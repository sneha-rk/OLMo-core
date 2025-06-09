#### ECC error

```
Traceback (most recent call last):
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/scripts/train.py", line 418, in <module>
    dist.init_process_group(
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/distributed/c10d_logger.py", line 95, in wrapper
    func_return = func(*args, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 1721, in init_process_group
    default_pg, _ = _new_process_group_helper(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2085, in _new_process_group_helper
    eager_backend.eager_connect_single_device(device_id)
torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.hpp:268, unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.21.5
ncclUnhandledCudaError: Call to CUDA function failed.
Last error:
Cuda failure 'uncorrectable ECC error encountered'
```
Encountered when using L40. Switching to A40 fixed this


#### incorrect data size / data unavailable
```
Traceback (most recent call last):
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/scripts/train.py", line 457, in <module>
    main(cfg)
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/scripts/train.py", line 397, in main
    trainer.fit()
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/train.py", line 1288, in fit
    for batch in self.train_loader:
                 ^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 708, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1480, in _next_data
    return self._process_data(data)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1505, in _process_data
    data.reraise()
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/_utils.py", line 733, in reraise
    raise exception
ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/utils/data/_utils/fetch.py", line 33, in fetch
    data.append(next(self.dataset_iter))
                ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/data/iterable_dataset.py", line 182, in <genexpr>
    return (self._get_dataset_item(int(idx)) for idx in indices)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/data/iterable_dataset.py", line 185, in _get_dataset_item
    item = self.dataset[idx]
           ~~~~~~~~~~~~^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/data/memmap_dataset.py", line 196, in __getitem__
    input_ids = self._read_chunk_from_memmap(self._memmap_paths[memmap_index], memmap_local_index)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/data/memmap_dataset.py", line 162, in _read_chunk_from_memmap
    buffer = get_bytes_range(path, bytes_start, num_bytes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/util.py", line 380, in get_bytes_range
    return _http_get_bytes_range(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/util.py", line 725, in _http_get_bytes_range
    raise ValueError(
ValueError: Failed to download 8192 bytes from http://olmo-data.org/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-038-00003.npy after 5 attempts.
```
```
Expected 8192 bytes, but got 12175. Retrying...
```
Response.content contains:
```
b'<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n    <link rel="icon" href="https://www.cloudflare.com/favicon.ico" />\n    <title>Internal Error</title>\n    <style>\n      body {\n        font-family: system-ui;\n        font-weight: 300;\n        font-size: 1.25rem;\n        color: #36393a;\n        display: flex;\n        align-items: center;\n        justify-content: center;\n      }\n      main {\n        max-width: 1200px;\n        margin-top: 120px;\n        display: flex;\n        flex-wrap: wrap;\n        align-items: center;\n        justify-content: center;\n      }\n      #text {\n        max-width: 60%;\n        margin-left: 1rem;\n        margin-right: 1rem;\n      }\n      main > section > div {\n        margin-bottom: 3.25rem;\n      }\n      svg {\n        margin-left: 2rem;\n      }\n      @keyframes tilt {\n        0% {\n          transform: rotate(0deg);\n        }\n        50% {\n          transform: rotate(3deg);\n        }\n        100% {\n          transform: rotate(0deg);\n        }\n      }\n      svg > #body {\n        animation: tilt 4s infinite;\n        transform-box: fill-box;\n        transform-origin: center;\n      }\n      h1 {\n        font-size: 3.75rem;\n        font-weight: 400;\n        margin-bottom: 0.5rem;\n      }\n      h3 {\n        font-size: 2rem;\n        font-weight: 400;\n        color: #92979b;\n        margin: 0;\n      }\n      p {\n        margin: 0;\n      }\n      #error-title {\n        font-size: 2rem;\n        margin-bottom: 1rem;\n      }\n    </style>\n  </head>\n\n  <body>\n    <main>\n      <section id="text">\n        <div>\n          <h1>Error 500</h1>\n          <h3>There was an internal error</h3>\n        </div>\n\n        <div>\n          <p id="error-title">We were not able to find this object</p>\n          <p>\n            Cloudflare was unable to retrieve the object at this time. Please\n            refresh or try again later.\n          </p>\n        </div>\n      </section>\n\n      <section>\n        <svg\n          xmlns="http://www.w3.org/2000/svg"\n          width="276"\n          height="385"\n          fill="none"\n          viewBox="0 0 276 385"\n        >\n          <g id="body">\n            <path\n              stroke="#0055DC"\n              stroke-miterlimit="10"\n              stroke-width="2"\n              d="M209.5 273.6c15.8-4.8 28-10.4 27.4-12.4-.6-2-13.9.3-29.6 5.1-15.8 4.8-28 10.4-27.4 12.4.6 2 13.9-.3 29.6-5.1Z"\n            />\n            <path\n              stroke="#3790E0"\n              stroke-miterlimit="10"\n              stroke-width="2"\n              d="M217 291.3c20.6-6.3 36.3-14.8 35-18.9-1.3-4.1-19-2.3-39.6 4s-36.3 14.8-35 18.9c1.3 4.1 19 2.3 39.6-4Z"\n              opacity=".6"\n            />\n            <path\n              stroke="#6ECCE5"\n              stroke-miterlimit="10"\n              stroke-width="2"\n              d="M227.5 311.8c26.7-8.2 47.1-19.2 45.4-24.5-1.6-5.3-24.6-3-51.4 5.2-26.7 8.2-47.1 19.2-45.4 24.5 1.6 5.3 24.6 3 51.4-5.2Z"\n              opacity=".6"\n            />\n            <path\n              fill="#C5EBF5"\n              stroke="#0055DC"\n              stroke-width="2"\n              d="M151.1 220c-2.8 5.4-9.4 7.5-14.8 4.7s-7.5-9.4-4.7-14.8 9.4-7.5 14.8-4.7 7.5 9.4 4.7 14.8Zm10.6-37.4c-.4 2.9-1.2 5.5-2.1 7.2-.4.9-.9 1.5-1.3 1.9-.4.4-.7.4-.7.4s-.3 0-.6-.6c-.3-.5-.5-1.2-.7-2.2-.4-1.9-.4-4.6 0-7.5.4-2.9 1.2-5.5 2.1-7.2.4-.9.9-1.5 1.3-1.9.4-.4.7-.4.7-.4s.3 0 .6.6c.3.5.5 1.2.7 2.2.4 1.9.4 4.6 0 7.5Z"\n            />\n            <path\n              stroke="#0055DC"\n              stroke-width="2"\n              d="M201.6 260.7c-11 0-21.7-9.1-29.8-23.1-8-13.9-13.2-32.3-13.2-50.1 0-17.8 5.2-28.5 13-34.8 7.9-6.3 18.6-8.4 29.9-8.4 11.3 0 22.1 2 29.9 8.3 7.8 6.3 13 17 13 34.9s-5.2 36.3-13.2 50.2c-8.1 13.9-18.8 23-29.8 23h.2Z"\n            />\n            <path\n              stroke="#C5EBF5"\n              stroke-width="12"\n              d="M201.6 255.7c-8.2 0-17.6-7.1-25.4-20.6-7.6-13.1-12.5-30.7-12.5-47.6 0-16.9 4.9-25.8 11.2-30.9 6.6-5.3 15.9-7.3 26.8-7.3s20.2 1.9 26.8 7.2c6.3 5 11.2 14.1 11.2 31s-4.9 34.5-12.5 47.7c-7.8 13.5-17.2 20.5-25.4 20.5h-.2Z"\n            />\n            <path\n              stroke="#0055DC"\n              stroke-width="2"\n              d="M201.6 260.7c-11 0-21.7-9.1-29.8-23.1-8-13.9-13.2-32.3-13.2-50.1 0-17.8 5.2-28.5 13-34.8 7.9-6.3 18.6-8.4 29.9-8.4 11.3 0 22.1 2 29.9 8.3 7.8 6.3 13 17 13 34.9s-5.2 36.3-13.2 50.2c-8.1 13.9-18.8 23-29.8 23h.2Z"\n            />\n            <path\n              fill="#C5EBF5"\n              stroke="#0055DC"\n              stroke-width="2"\n              d="M265.7 145.3c2.5 5.6 0 12.1-5.6 14.5-5.6 2.5-12.1 0-14.5-5.6-2.5-5.6 0-12.1 5.6-14.5 5.6-2.5 12.1 0 14.5 5.6Zm-13.8 21.5c.5 1.2 0 2.7-1.2 3.2s-2.7 0-3.2-1.2 0-2.7 1.2-3.2 2.7 0 3.2 1.2Zm-99.2 27.4c.5 1.2 0 2.7-1.2 3.2s-2.7 0-3.2-1.2 0-2.7 1.2-3.2 2.7 0 3.2 1.2Zm88.8-22.2s.3 0 .7.5c.4.4.7 1.1 1 2.1.6 1.9 1.1 4.5 1.1 7.5s-.4 5.6-1 7.5c-.3.9-.7 1.6-1 2.1-.4.4-.6.5-.7.5 0 0-.3 0-.7-.5-.3-.4-.7-1.1-1-2-.6-1.9-1.1-4.5-1.1-7.5s.4-5.6 1-7.5c.3-.9.7-1.6 1-2.1.4-.4.6-.5.7-.5v-.1Z"\n            />\n            <path\n              fill="#C5EBF5"\n              d="M237.6 68.9H166c-5.2 0-9.4 4.2-9.4 9.4v49.2c0 5.2 4.2 9.4 9.4 9.4h71.6c5.2 0 9.4-4.2 9.4-9.4V78.3c0-5.2-4.2-9.4-9.4-9.4Zm1.1 54.9c0 3.7-3 6.8-6.8 6.8h-60.1c-3.7 0-6.8-3-6.8-6.8v-40c0-3.7 3-6.8 6.8-6.8h60.1c3.7 0 6.8 3 6.8 6.8v40Z"\n            />\n            <path\n              fill="#6ECCE5"\n              d="M221 83.5c-8.6 0-15.6 6.9-15.7 15.6h-5c0-8.6-7.1-15.5-15.7-15.6-8.7 0-15.7 7-15.7 15.7s7 15.7 15.7 15.7c8.1 0 14.7-6.1 15.6-13.9h5.2c.9 7.8 7.5 13.9 15.6 13.9 8.7 0 15.7-7 15.7-15.7s-7-15.7-15.7-15.7Zm-26.7 25.4c-2.5 2.5-5.9 4-9.7 4-3.8 0-7.2-1.5-9.7-4-2.5-2.5-4-5.9-4-9.7 0-3.8 1.5-7.2 4-9.7 2.5-2.5 5.9-4 9.7-4 3.8 0 7.2 1.5 9.7 4 2.5 2.5 4 5.9 4 9.7 0 3.8-1.5 7.2-4 9.7Zm36.4 0c-2.5 2.5-5.9 4-9.7 4-3.8 0-7.2-1.5-9.7-4-2.5-2.5-4-5.9-4-9.7 0-3.8 1.5-7.2 4-9.7 2.5-2.5 5.9-4 9.7-4 3.8 0 7.2 1.5 9.7 4 2.5 2.5 4 5.9 4 9.7 0 3.8-1.5 7.2-4 9.7Z"\n            />\n            <path\n              fill="#0055DC"\n              d="M193.2 124.1c-1.1 0-2.1-.2-3.1-.7-1-.5-1.9-1-2.7-1.5-.8-.5-1.7-.8-2.4-.8s-.9 0-1.2.3c-.4.2-.8.5-1.1.9l-1.9-2c.6-.6 1.3-1.1 2-1.4.7-.3 1.5-.5 2.2-.5.7 0 2.1.3 3 .8 1 .5 2 1 2.8 1.5.9.5 1.7.7 2.4.7s1.9-.3 2.9-1 2-1.4 3.1-2.1c1.1-.7 2.2-1 3.5-1 1.3 0 2.3.3 3.4 1s2.1 1.4 3.2 2.1c1 .7 2 1 2.9 1 .9 0 1.5-.2 2.4-.7.9-.5 1.8-1 2.8-1.5 1-.5 2-.8 3.1-.8s1.5.2 2.2.5c.7.3 1.4.8 2 1.4l-1.9 2c-.4-.4-.8-.7-1.1-.9-.4-.2-.8-.3-1.2-.3-.8 0-1.6.3-2.5.8-.8.5-1.8 1-2.7 1.5-1 .5-2 .7-3.1.7s-2.5-.3-3.6-1-2.1-1.4-3.1-2.1c-1-.7-1.9-1-2.7-1-.8 0-1.8.3-2.7 1-1 .7-2 1.4-3.1 2.1-1.1.7-2.3 1-3.6 1h-.2Zm-12.9-16.7c1.5 1.3 3.5 2 5.9 2 2.4 0 4.9-.9 6.6-2.7 1.7-1.8 2.5-4.1 2.5-7 0-2.9-.5-4-1.4-5.6-.9-1.6-2.2-2.9-3.8-3.9-1.6-.9-3.4-1.4-5.4-1.4-2 0-3.5.4-5 1.3-1.5.8-2.8 2-3.8 3.5s-1.5 3.2-1.6 5l1 .2c.5-2.4 1.7-4.3 3.5-5.6 1.8-1.3 3.8-2 5.9-2 2.1 0 4.4.8 6 2.3 1.6 1.5 2.4 3.6 2.4 6.3 0 2.7-.6 3.8-1.8 5.2-1.2 1.4-2.9 2.1-5.1 2.1-2.2 0-3.3-.5-4.4-1.4-1.1-.9-1.6-2.1-1.6-3.6s.4-2.4 1.3-3.4c.8-1 1.9-1.5 3.1-1.5 1.2 0 1.5.2 2 .7.5.4.8.9.8 1.6 0 .7-.2 1.1-.6 1.5-.4.4-.9.6-1.5.6s-1-.1-1.7-.4l-.6 1.8c.6.2 1 .4 1.3.5.4 0 .7.1 1 .1 1.2 0 2.1-.4 2.9-1.1.8-.8 1.2-1.8 1.2-2.9 0-1.1-.5-2.3-1.4-3.1-.9-.8-2-1.2-3.4-1.2s-3.4.7-4.7 2.1c-1.3 1.4-1.9 3-1.9 4.9 0 1.9.8 3.9 2.3 5.2v-.1Zm34.5 1.5c1.6.9 3.4 1.4 5.4 1.4 2 0 3.5-.4 5-1.3s2.8-2 3.8-3.5 1.5-3.2 1.6-5l-1-.2c-.5 2.4-1.6 4.3-3.5 5.6-1.8 1.3-3.7 2-5.9 2s-4.4-.8-6-2.3c-1.6-1.5-2.4-3.6-2.4-6.3 0-2.7.6-3.8 1.8-5.2 1.2-1.4 2.9-2.1 5.1-2.1 2.2 0 3.3.5 4.3 1.4 1.1.9 1.6 2.1 1.6 3.6s-.4 2.4-1.3 3.4c-.8 1-1.9 1.4-3.1 1.4-1.2 0-1.5-.2-2-.6-.5-.4-.8-1-.8-1.6 0-.6.2-1.1.6-1.5.4-.4.9-.6 1.5-.6s1 .1 1.7.4l.6-1.8c-.6-.2-1-.4-1.4-.4-.3 0-.7-.1-1-.1-1.2 0-2.1.4-2.9 1.2-.8.8-1.1 1.7-1.1 2.9s.4 2.3 1.3 3.2c.9.8 2.1 1.2 3.5 1.2s3.4-.7 4.7-2.1c1.3-1.4 1.9-3 1.9-4.9 0-1.9-.8-3.9-2.3-5.2-1.5-1.3-3.5-2-5.9-2-2.4 0-4.9.9-6.6 2.7-1.7 1.8-2.5 4.1-2.5 7 0 2.9.5 4 1.4 5.6.9 1.6 2.2 2.9 3.8 3.9l.1-.2ZM1.3 348.6l1 .1-1-.1ZM250.5 88.4l-.2 1 .2-1Z"\n            />\n            <path\n              fill="#0055DC"\n              d="M251.2 92H248V78.3c0-5.8-4.7-10.4-10.4-10.4H166c-5.8 0-10.4 4.7-10.4 10.4V92h-3.1v21.9h3.1v13.7c0 5.8 4.7 10.4 10.4 10.4h71.6c5.8 0 10.4-4.7 10.4-10.4v-13.7h3.2V92Zm-5.2 35.6c0 2.3-.9 4.4-2.5 6-1.5 1.5-3.6 2.5-6 2.5h-71.6c-2.3 0-4.4-.9-6-2.5-1.5-1.5-2.5-3.6-2.5-6V78.4c0-2.3.9-4.4 2.5-6 1.5-1.5 3.6-2.5 6-2.5h71.6c2.3 0 4.4.9 6 2.5 1.5 1.5 2.5 3.6 2.5 6v49.2Z"\n            />\n            <path\n              fill="#C5EBF5"\n              d="m252.9 79.6-1.2-.3 6.6-61-55.5-15L179 59.7l-1.1-.3c-1.3-.3-2.6.4-2.9 1.7l-1.4 5.2c-.3 1.3.4 2.6 1.7 2.9l75 20.1c1.3.3 2.6-.4 2.9-1.7l1.4-5.2c.3-1.3-.4-2.6-1.7-2.9v.1Z"\n            />\n            <path\n              fill="#0055DC"\n              d="M253.2 78.6h-.4c0-.1 6.5-60.3 6.5-60.3 0-.5-.3-.9-.7-1.1L203 2.4c-.5-.1-1 .1-1.2.6l-23.4 55.6h-.3c-.3-.2-.6-.2-.9-.2-1.5 0-2.9 1-3.3 2.5l-1.4 5.2c0 .3-.1.6-.1.9 0 1.5 1 2.9 2.5 3.3l75 20.1c.3 0 .6.1.9.1 1.5 0 2.9-1 3.3-2.5l1.4-5.2c0-.3.1-.6.1-.9 0-1.5-1-2.9-2.5-3.3h.1ZM203.4 4.5l53.8 14.4-6.4 59L180.4 59l23-54.5Zm50.3 77.7-1.4 5.2c-.2.6-.7 1-1.4 1h-.4l-75-20.1c-.6-.2-1-.7-1-1.4v-.4l1.4-5.2c.2-.6.7-1 1.4-1h.4l1.9.5 71 19 2 .5c.6.2 1 .8 1 1.4v.4l.1.1Z"\n            />\n          </g>\n          <path\n            fill="#E2F5FA"\n            d="M140 385c69.036 0 125-14.551 125-32.5 0-17.949-55.964-32.5-125-32.5S15 334.551 15 352.5c0 17.949 55.964 32.5 125 32.5Z"\n          />\n          <path\n            fill="#C5EBF5"\n            d="M195.8 281H137l2.1-16.2h.9c1.3 0 2.4-1.1 2.4-2.4v-5.7c0-1.3-1.1-2.4-2.4-2.4h-1.9l1-7.9h.9c1.3 0 2.4-1.1 2.4-2.4v-5.7c0-1.3-1.1-2.4-2.4-2.4H66.8c-1.3 0-2.4 1.1-2.4 2.4v5.7c0 1.3 1.1 2.4 2.4 2.4h1l1.1 7.9h-2.1c-1.3 0-2.4 1.1-2.4 2.4v5.7c0 1.3 1.1 2.4 2.4 2.4h1l4.3 29.8-48-14.4v-.8c.4-1.3-.5-2.6-1.8-2.8l-5.4-.9c-1.3-.2-2.6.7-2.8 2l-11.8 71c-.2 1.3.7 2.6 2 2.8l5.4.9c1.3.2 2.6-.7 2.8-2l.2-.9 61.9 1.4 4-23.9h22.6l4.5 31.2h33.2l.8-6.3h47.6l7.7-60.7h.8c1.3 0 2.4-1.1 2.4-2.4v-5.4c0-1.3-1.1-2.4-2.4-2.4Z"\n          />\n          <path\n            fill="#6ECCE5"\n            d="m100.3 328-.3-2H79.7l-.3 2h20.9Zm39.6-80.6c1.9 0 3.4-1.5 3.4-3.4v-5.7c0-1.9-1.5-3.4-3.4-3.4H66.8c-1.9 0-3.4 1.5-3.4 3.4v5.7c0 1.9 1.5 3.4 3.4 3.4h.1l.9 5.9h-1c-1.9 0-3.4 1.5-3.4 3.4v5.7c0 1.9 1.5 3.4 3.4 3.4h.1l3.9 27.4 2.1.6-4-28.1h68.9l-1.8 14.2h2l1.8-14.2c1.9 0 3.4-1.5 3.4-3.4v-5.7c0-1.9-1.5-3.4-3.4-3.4h-.8l.8-5.9.1.1Zm0 7.9c.8 0 1.4.6 1.4 1.4v5.7c0 .7-.5 1.3-1.2 1.4H66.6c-.7 0-1.3-.7-1.3-1.4v-5.7c0-.8.6-1.4 1.4-1.4h73.2Zm-71-7.9h68.9l-.8 5.9H69.8l-.9-5.9Zm69.1-2H66.6c-.7 0-1.3-.7-1.3-1.4v-5.7c0-.8.6-1.4 1.4-1.4h73.1c.8 0 1.4.6 1.4 1.4v5.7c0 .7-.5 1.3-1.2 1.4h-2Z"\n          />\n          <path\n            fill="#0055DC"\n            d="m83.7 297-10.6-3.2-2.1-.6-45.6-13.7v-.5c0-1.6-1.2-3.1-2.8-3.4l-5.4-.9h-.6c-1.6 0-3.1 1.2-3.4 2.8L1.3 348.6v.6c0 1.6 1.2 3.1 2.8 3.4l5.4.9h.6c1.6 0 3.1-1.2 3.4-2.8l61.1 1.3c.5 0 .9-.3 1-.8l3.8-23 .3-2 4.6-27.8c0-.5-.2-1-.7-1.1l.1-.3Zm-71.9 51.5-.3 1.8v.2c-.2.6-.7 1-1.3 1H10l-5.4-.9c-.7-.1-1.2-.7-1.2-1.4v-.2l11.8-71c.1-.7.7-1.2 1.4-1.2h.2l5.4.9c.7.1 1.1.7 1.2 1.3v.2l-.3 1.7-11.2 67.5-.1.1ZM77.7 326l-.3 2-3.6 21.9-59.9-1.3 11.1-67 46.3 13.8 2.1.6 8.9 2.7-4.5 27.3h-.1Zm60.3 31.2h-31.5l-4.2-29.2h-2l4.4 30.3c0 .5.5.9 1 .9h33.2c.5 0 .9-.4 1-.9l.7-5.4h-2l-.5 4.3h-.1Z"\n          />\n          <path\n            fill="#0055DC"\n            d="m138.8 350.8-.3 2h2l.3-2h-2ZM102 326h-2l.3 2h2l-.3-2Z"\n          />\n          <path\n            fill="#0055DC"\n            d="M195.8 280h-71.2c-1.9 0-3.4 1.5-3.4 3.4v5.4c0 1.9 1.5 3.4 3.4 3.4l3 21h-26.8c-1.9 0-3.4 1.5-3.4 3.4v.8c0 1.3.7 2.3 1.7 2.9l.8 5.7h2l-.7-5.2h41.3l-3.8 30.1h2l3.9-30.6c1-.6 1.7-1.7 1.7-2.9v-.8c0-1.9-1.5-3.4-3.4-3.4h-13.2l-3-21h67.1l-7.4 58.7h-45.6l-.3 2h46.7c.5 0 .9-.4 1-.9l7.6-59.8c1.8 0 3.3-1.5 3.3-3.4v-5.4c0-1.9-1.5-3.4-3.4-3.4h.1ZM143 315.2c.7 0 1.3.6 1.4 1.3v1c0 .3-.1.6-.3.8 0 .1-.1.1-.1.1-.3.3-.6.4-1 .4h-42c-.8 0-1.4-.6-1.4-1.4v-1c0-.7.7-1.3 1.4-1.3h42v.1Zm54.3-26.4c0 .7-.5 1.2-1.1 1.4h-71.9c-.6-.1-1.1-.7-1.1-1.4v-5.4c0-.8.6-1.4 1.4-1.4h71.1c.8 0 1.4.6 1.4 1.4v5.4h.2Z"\n          />\n        </svg>\n      </section>\n    </main>\n  </body>\n</html>\n'
```

Not sure what caused this 

#### NCCL / distbackend errors
```

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:43 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x1473768991b6 in /gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x147376842a76 in /gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #2: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x118 (0x147376987918 in /gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/lib/libc10_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x56 (0x147377bd5556 in /gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0xa0 (0x147377be28c0 in /gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::watchdogHandler() + 0x617 (0x147377be4557 in /gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x147377be56ed in /gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #7: <unknown function> + 0x145c0 (0x1473c0c685c0 in /gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/lib/libtorch.so)
frame #8: <unknown function> + 0x81ca (0x1473d7c071ca in /lib64/libpthread.so.0)
frame #9: clone + 0x43 (0x1473d70d88d3 in /lib64/libc.so.6)

terminate called after throwing an instance of 'c10::DistBackendError'
  what():  [PG ID 0 PG GUID 0(default_pg) Rank 0] Process group watchdog thread terminated with exception: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

Relaunched with CUDA_LAUNCH_BLOCKING=1

It ended up being actually the error below (embedding/vocab mismatch)

#### WandB initialization timeout
```
wandb: ERROR Run initialization has timed out after 90.0 sec. Please try increasing the timeout with the `init_timeout` setting: `wandb.init(settings=wandb.Settings(init_timeout=120))`.
Traceback (most recent call last):
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/asyncio/tasks.py", line 520, in wait_for
    return await fut
           ^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/asyncio/locks.py", line 212, in wait
    await fut
asyncio.exceptions.CancelledError

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/mailbox/response_handle.py", line 109, in wait_async
    await asyncio.wait_for(evt.wait(), timeout=timeout)
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/asyncio/tasks.py", line 519, in wait_for
    async with timeouts.timeout(timeout):
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/asyncio/timeouts.py", line 115, in __aexit__
    raise TimeoutError from exc_val
TimeoutError

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/wandb_init.py", line 941, in init
    result = wait_with_progress(
             ^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/mailbox/wait_with_progress.py", line 24, in wait_with_progress
    return wait_all_with_progress(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/mailbox/wait_with_progress.py", line 87, in wait_all_with_progress
    return asyncio_compat.run(progress_loop_with_timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/lib/asyncio_compat.py", line 30, in run
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/concurrent/futures/_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/concurrent/futures/thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/lib/asyncio_compat.py", line 74, in run
    return asyncio.run(self._run_or_cancel(fn))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/asyncio/base_events.py", line 691, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/lib/asyncio_compat.py", line 98, in _run_or_cancel
    return fn_task.result()
           ^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/mailbox/wait_with_progress.py", line 82, in progress_loop_with_timeout
    return await _wait_handles_async(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/mailbox/wait_with_progress.py", line 130, in _wait_handles_async
    async with asyncio_compat.open_task_group() as task_group:
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/contextlib.py", line 217, in __aexit__
    await anext(self.gen)
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/lib/asyncio_compat.py", line 190, in open_task_group
    await task_group._wait_all()
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/lib/asyncio_compat.py", line 159, in _wait_all
    raise exc
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/mailbox/wait_with_progress.py", line 128, in wait_single
    results[index] = await handle.wait_async(timeout=timeout)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/mailbox/mailbox_handle.py", line 126, in wait_async
    response = await self._handle.wait_async(timeout=timeout)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/mailbox/response_handle.py", line 118, in wait_async
    raise TimeoutError(
TimeoutError: Timed out waiting for response on uwpixizj5au8

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/scripts/train.py", line 457, in <module>
    main(cfg)
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/scripts/train.py", line 118, in main
    wandb.init(
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/wandb_init.py", line 1482, in init
    wandb._sentry.reraise(e)
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/analytics/sentry.py", line 156, in reraise
    raise exc.with_traceback(sys.exc_info()[2])
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/wandb_init.py", line 1468, in init
    return wi.init(run_settings, run_config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/wandb/sdk/wandb_init.py", line 954, in init
    raise CommError(
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec. Please try increasing the timeout with the `init_timeout` setting: `wandb.init(settings=wandb.Settings(init_timeout=120))`.
[rank0]:[W325 14:57:06.449525020 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
srun: error: g3060: task 0: Exited with exit code 1
srun: Terminating StepId=25036072.0
slurmstepd: error: *** STEP 25036072.0 ON g3060 CANCELLED AT 2025-03-25T14:57:08 ***
srun: error: g3060: tasks 1-3: Killed
srun: Force Terminated StepId=25036072.0
```
Happens if the wandb run_id is not unique. Adding datetime to the run id string fixes this

#### No Kernel available

```
Traceback (most recent call last):
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/scripts/train.py", line 459, in <module>
    main(cfg)
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/scripts/train.py", line 399, in main
    trainer.fit()
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/train.py", line 1319, in fit
    metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/train.py", line 923, in train_step
    ce_batch_loss, z_batch_loss, lb_batch_loss, moe_z_batch_loss, expert_assignments = self.train_batch(batch)
                                                                                       ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/train.py", line 864, in train_batch
    loss, ce_loss, z_loss = self.train_micro_batch(micro_batch, batch_size_in_tokens)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/train.py", line 791, in train_micro_batch
    ce_loss, z_loss, logits = self.model_forward(
                              ^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/train.py", line 764, in model_forward
    logits = self.dist_model(
             ^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 864, in forward
    output = self._fsdp_wrapped_module(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/model.py", line 1564, in forward
    x, cache = block(
               ^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 864, in forward
    output = self._fsdp_wrapped_module(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/olmoe/olmo/model.py", line 830, in forward
    return og_x + self.dropout(self.ffn(x)), cache
                               ^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/megablocks/megablocks/layers/moe.py", line 529, in forward
    out = self.experts(x, scores, logits, expert_weights, top_experts)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/megablocks/megablocks/layers/moe.py", line 449, in forward
    x, tokens_per_expert = self.forward_fn(x, expert_weights, top_experts)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/megablocks/megablocks/layers/dmoe.py", line 286, in forward_once
    return self.grouped_forward_once(x, expert_weights, top_experts)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/megablocks/megablocks/layers/dmoe.py", line 246, in grouped_forward_once
    indices, bin_ids, bins, tokens_per_expert = (self.indices_and_bins(top_experts))
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/megablocks/megablocks/layers/moe.py", line 178, in indices_and_bins
    output = ops.sort(top_expert, self.sort_end_bit)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gscratch/zlab/margsli/miniforge3/envs/moe/lib/python3.12/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mmfs1/gscratch/zlab/margsli/gitfiles/megablocks/megablocks/ops/sort.py", line 34, in forward
    ops.sort(x, end_bit, x_out, iota_out)
RuntimeError: no kernel image is available for execution on the device
```