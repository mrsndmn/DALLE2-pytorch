import os
import webdataset as wds
import torch
from torch.utils.data import DataLoader
import numpy as np
import fsspec
import shutil

def get_shard(filename):
    """
    Filenames with shards in them have a consistent structure that we can take advantage of
    Standard structure: path/to/file/prefix_string_00001.ext
    """
    try:
        return filename.split("_")[-1].split(".")[0]
    except ValueError:
        raise RuntimeError(f"Could not find shard for filename {filename}")

def get_example_file(fs, path, file_format):
    """
    Given a file system and a file extension, return the example file
    """
    return fs.glob(os.path.join(path, f"*.{file_format}"))[0]

def embedding_inserter(samples, embeddings_url, index_width, sample_key='npy', handler=wds.handlers.reraise_exception):
    """Given a datum of {"__key__": str, "__url__": str, ...} adds the cooresponding embedding and yields"""
    previous_tar_url = None
    current_embeddings = None
    # Get a reference to an abstract file system where the embeddings are stored
    embeddings_fs, embeddings_path = fsspec.core.url_to_fs(embeddings_url)
    example_embedding_file = get_example_file(embeddings_fs, embeddings_path, "npy")
    example_embedding_shard = get_shard(example_embedding_file)
    emb_shard_width = len(example_embedding_shard)
    # Easier to get the basename without the shard once than search through for the correct file every time
    embedding_file_basename = '_'.join(example_embedding_file.split("_")[:-1]) + "_"

    def load_corresponding_embeds(tar_url):
      """Finds and reads the npy files that contains embeddings for the given webdataset tar"""
      shard = int(tar_url.split("/")[-1].split(".")[0])
      embedding_url = embedding_file_basename + str(shard).zfill(emb_shard_width) + '.npy'
      with embeddings_fs.open(embedding_url) as f:
        data = np.load(f)
      return torch.from_numpy(data)

    for sample in samples:
        try:
            tar_url = sample["__url__"]
            key = sample["__key__"]
            if tar_url != previous_tar_url:
                # If the tar changed, we need to download new embeddings
                # This means if we shuffle before inserting it will load many more files than we expect and be very inefficient.
                previous_tar_url = tar_url
                current_embeddings = load_corresponding_embeds(tar_url)

            embedding_index = int(key[-index_width:])
            embedding = current_embeddings[embedding_index]
            # We need to check if this sample is nonzero. If it is, this embedding is not valid and we should continue to the next loop
            if torch.count_nonzero(embedding) == 0:
                raise RuntimeError(f"Webdataset had a sample, but no embedding was found. ImgShard: {key[:-index_width]} - Index: {key[-index_width:]}")
            sample[sample_key] = embedding
            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break
insert_embedding = wds.filters.pipelinefilter(embedding_inserter)

def unassociated_shard_skipper(tarfiles, embeddings_url, handler=wds.handlers.reraise_exception):
    """Finds if the is a corresponding embedding for the tarfile at { url: [URL] }"""
    embeddings_fs, embeddings_path = fsspec.core.url_to_fs(embeddings_url)
    embedding_files = embeddings_fs.ls(embeddings_path)
    get_embedding_shard = lambda embedding_file: int(embedding_file.split("_")[-1].split(".")[0])
    embedding_shards = set([get_embedding_shard(filename) for filename in embedding_files])  # Sets have O(1) check for member

    get_tar_shard = lambda tar_file: int(tar_file.split("/")[-1].split(".")[0])
    for tarfile in tarfiles:
        try:
            webdataset_shard = get_tar_shard(tarfile["url"])
            # If this shard has an associated embeddings file, we pass it through. Otherwise we iterate until we do have one
            if webdataset_shard in embedding_shards:
                yield tarfile
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break
skip_unassociated_shards = wds.filters.pipelinefilter(unassociated_shard_skipper)


def join_embeddings(samples, handler=wds.handlers.reraise_exception):
    """
    Takes the img_emb and text_emb keys and turns them into one key "emb": { "text": text_emb, "img": img_emb }
    either or both of text_emb and img_emb may not be in the sample so we only add the ones that exist
    """
    for sample in samples:
        try:
            if 'text_emb' in sample:
                sample['emb']['text'] = sample['text_emb']
            if 'img_emb' in sample:
                sample['emb']['img'] = sample['img_emb']
            if 'audio_emb' in sample:
                sample['emb']['audio'] = sample.pop('audio_emb')

            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break

def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """
    for sample in samples:

        if 'melspectrogram.npy' not in sample:
            print("'melspectrogram.npy' not in sample", list(sample.keys()))
            continue
        if 'audio_emb.npy' not in sample:
            print("'audio_emb.npy' not in sample", list(sample.keys()))
            continue

        sample['audio_emb'] = sample.pop('audio_emb.npy')
        sample['audio_melspec'] = sample.pop('melspectrogram.npy')

        audio_max_len = 512
        if sample['audio_melspec'].shape[-1] < audio_max_len:
            print("too little item:", sample['audio_melspec'].shape)
            continue

        sample['audio_melspec'] = sample['audio_melspec'][:, : , :audio_max_len]

        try:
            for key in required_keys:
                assert key in sample, f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}"
            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break
key_verifier = wds.filters.pipelinefilter(verify_keys)

class AudioEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            audio_embeddings_url=None,
            audio_melspectrogram_url=None,
            text_embedding_folder_url=None,
            index_width=None,
            audio_mepspec_preproc=None,
            extra_keys=[],
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True
    ):
        """
        Modeled directly off of the WebDataset constructor

        :param urls: A url pointing to the tar files of the webdataset formatted as /path/to/webdataset/{0000..9999}.tar
        :param embedding_folder_url: Required if webdataset does not contain embeddings. A url pointing to the npy files of the embeddings. Should have the same number of shards as the webdataset.
            Webdataset image keys should align with the index of the embedding. This means missing image indices must have a corresponding embedding of all zeros.
        :param index_width: The number of digits in the index. This is used to align the embedding index with the image index.
            For example, if a file in the webdataset shard 3 is named 0003039.jpg, we know the shard is 4 digits and the last 3 digits are the index_width.
        :param audio_mepspec_preproc: This function is run on the img before it is batched and returned. Useful for data augmentation or converting to torch tensor.
        :param handler: A webdataset handler.
        :param resample: If true, resample webdataset shards with replacement. You need to set your own epoch size if this is true since it will resample infinitely.
        :param shuffle_shards: If true, shuffle the shards before resampling. This cannot be true if resample is true.


        """
        super().__init__()
        keys = ["audio_melspec", "audio_emb"] + extra_keys

        # print("urls", urls)

        self.key_map = {key: i for i, key in enumerate(keys)}
        self.resampling = resample
        self.audio_mepspec_preproc = audio_mepspec_preproc
        # If s3, check if s3fs is installed and s3cmd is installed and check if the data is piped instead of straight up
        if (isinstance(urls, str) and "s3:" in urls) or (isinstance(urls, list) and any(["s3:" in url for url in urls])):
            # Then this has an s3 link for the webdataset and we need extra packages
            if shutil.which("s3cmd") is None:
                raise RuntimeError("s3cmd is required for s3 webdataset")
        if (audio_melspectrogram_url is not None and "s3:" in audio_melspectrogram_url) or (text_embedding_folder_url is not None and "s3:" in text_embedding_folder_url):
            # Then the embeddings are being loaded from s3 and fsspec requires s3fs
            try:
                import s3fs
            except ImportError:
                raise RuntimeError("s3fs is required to load embeddings from s3")
        # Add the shardList and randomize or resample if requested
        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        # if text_embedding_folder_url is not None:
        #     self.append(skip_unassociated_shards(embeddings_url=text_embedding_folder_url, handler=handler))
        # if audio_melspectrogram_url is not None:
        #     self.append(skip_unassociated_shards(embeddings_url=audio_melspectrogram_url, handler=handler))

        self.append(wds.tarfile_to_samples(handler=handler))

        # if audio_melspectrogram_url is not None:
        #     # Then we are loading image embeddings for a remote source
        #     assert index_width is not None, "Reading embeddings separately requires index width length to be given"
        #     self.append(insert_embedding(embeddings_url=audio_melspectrogram_url, index_width=index_width, sample_key='audio_melspec', handler=handler))
        # if audio_embeddings_url is not None:
        #     # Then we are loading image embeddings for a remote source
        #     assert index_width is not None, "Reading embeddings separately requires index width length to be given"
        #     self.append(insert_embedding(embeddings_url=audio_embeddings_url, index_width=index_width, sample_key='audio_emb', handler=handler))
        # if text_embedding_folder_url is not None:
        #     # Then we are loading image embeddings for a remote source
        #     assert index_width is not None, "Reading embeddings separately requires index width length to be given"
        #     self.append(insert_embedding(embeddings_url=text_embedding_folder_url, index_width=index_width, sample_key='text_emb', handler=handler))
        self.append(wds.decode(handler=handler))

        # self.append(join_embeddings)
        self.append(key_verifier(required_keys=keys, handler=handler))
        # Apply preprocessing
        self.append(wds.map(self.preproc))
        # self.append(wds.to_tuple(*keys))

    def preproc(self, sample):
        """Applies the preprocessing for images"""
        if self.audio_mepspec_preproc is not None:
            sample["audio_melspec"] = self.audio_mepspec_preproc(sample["audio_melspec"])
        return sample

def create_audio_embedding_dataloader(
    tar_url,
    num_workers,
    batch_size,
    # img_embeddings_url=None,
    # text_embeddings_url=None,
    audio_embeddings_url=None,
    audio_melspectrogram_url=None,
    index_width=None,
    shuffle_num = None,
    shuffle_shards = True,
    resample_shards = False,
    audio_preproc=None,
    extra_keys=[],
    handler=wds.handlers.reraise_exception#warn_and_continue
):
    """
    Convenience function to create an image embedding dataseta and dataloader in one line

    :param tar_url: A url pointing to the tar files of the webdataset formatted as /path/to/webdataset/{0000..9999}.tar
    :param num_workers: The number of workers to use for the dataloader
    :param batch_size: The batch size to use for the dataloader
    :param embeddings_url: Required if webdataset does not contain embeddings. A url pointing to the npy files of the embeddings. Should have the same number of shards as the webdataset.
        Webdataset image keys should align with the index of the embedding. This means missing image indices must have a corresponding embedding of all zeros.
    :param index_width: The number of digits in the index. This is used to align the embedding index with the image index.
            For example, if a file in the webdataset shard 3 is named 0003039.jpg, we know the shard is 4 digits and the last 3 digits are the index_width.
    :param shuffle_num: If not None, shuffle the dataset with this size buffer after sampling.
    :param shuffle_shards: If true, shuffle the shards before sampling. This cannot be true if resample is true.
    :param resample_shards: If true, resample webdataset shards with replacement. You need to set your own epoch size if this is true since it will resample infinitely.
    :param handler: A webdataset handler.
    """
    ds = AudioEmbeddingDataset(
        tar_url,
        audio_embeddings_url=audio_embeddings_url,
        audio_melspectrogram_url=audio_melspectrogram_url,
        # img_embedding_folder_url=img_embeddings_url,
        # text_embedding_folder_url=text_embeddings_url,
        index_width=index_width,
        shuffle_shards=shuffle_shards,
        resample=resample_shards,
        extra_keys=extra_keys,
        audio_mepspec_preproc=audio_preproc,
        handler=handler
    )

    # print('ds[0]', next(iter(ds)))

    if shuffle_num is not None and shuffle_num > 0:
        ds.shuffle(1000)

    def my_collate_fn(batch):


        # audio_melspec_stacked = torch.stack([ x['audio_melspec'] for x in batch ]).permute(0, 2, 1, 3)
        audio_melspec_stacked = torch.stack([ x['audio_melspec'] for x in batch ]).permute(0, 2, 3, 1)
        print("audio_melspec_stacked", audio_melspec_stacked.shape)

        return {
            "audio_emb": torch.stack([ torch.tensor(x['audio_emb']) for x in batch ])[:, 0, :],
            "audio_melspec": audio_melspec_stacked,
            "txt":[ x['txt'] for x in batch ],
        }

    return DataLoader(
        ds,
        num_workers=num_workers,
        batch_size=batch_size,
        prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
        pin_memory=True,
        shuffle=False,
        collate_fn=my_collate_fn,
        drop_last=True,
    )
