import json
import urllib.request


def query_llm(
    prompt,
    address="127.0.0.1",
    port=11434,
    model="gemma3:4b",
    endpoint="/api/chat",
    temperature=0.6,
    max_tokens=40,
    timeout=10,
):
    """
    Sends a prompt to a large language model server and returns the response.

    Parameters
    ----------
    prompt      : str
                  Input text prompt.
    address     : str
                  Address of the model server.
    port        : int
                  Port number of the model server.
    model       : str
                  Name of the model.
    endpoint    : str
                  API endpoint.
    temperature : float
                  Sampling temperature.
    max_tokens  : int
                  Maximum number of generated tokens.
    timeout     : int
                  Request timeout in seconds.

    Returns
    -------
    answer      : str
                  Generated response from the model.
    """
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")

    url = "http://{}:{}{}".format(
        address,
        port,
        endpoint,
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        data = json.loads(response.read().decode("utf-8"))

    if "message" not in data:
        raise ValueError("Invalid response format from model server.")

    if "content" not in data["message"]:
        raise ValueError("Missing content field in model response.")

    answer = data["message"]["content"].strip()

    return answer
