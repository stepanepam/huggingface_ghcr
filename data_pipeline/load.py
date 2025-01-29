from deprecated import deprecated


def load_fairy_tale(path_to_text: str):
    with open(path_to_text) as f:
        fairy_tale = f.read()
    return fairy_tale
