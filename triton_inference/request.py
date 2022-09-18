from pprint import pprint

import fire
import numpy as np
import tritonclient.http as httpclient


def test_infer(
    triton_client,
    model_name,
    input0_data,
    headers=None,
    request_compression_algorithm=None,
    response_compression_algorithm=None,
):
    inputs = []
    inputs.append(httpclient.InferInput('INPUT0', [4], "FP32"))
    inputs[0].set_data_from_numpy(input0_data, binary_data=True)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))

    # query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(
        model_name,
        inputs,
        outputs=outputs,
        #   query_params=query_params,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)

    return results


def main(url: str):
    triton_client = httpclient.InferenceServerClient(
        url=url,
        verbose=True,
    )

    input_data = np.array([1, 2, 3, 4], dtype=np.float32)

    result = test_infer(
        triton_client,
        'reverse',
        input_data,
    )
    pprint(result.get_response())
    output = result.as_numpy('OUTPUT0')
    print(output)


if __name__ == '__main__':
    fire.Fire(main)
