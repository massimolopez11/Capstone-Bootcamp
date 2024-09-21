"""Tools for Project"""

from PIL import Image


def is_valid_image(file_name: str) -> bool:
    """Checks if the image is valid according to Pillow (PIL) library. Is used when the
    user has files other than image in the uploaded zip folder.

    Args:
        filename: the path name relative to where the function is running.

    Returns:
        True if the file is of image type and False otherwise.

    """
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False
