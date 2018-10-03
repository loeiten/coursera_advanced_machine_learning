"""
Modified from the un-compiled version of get_data.pyc using `uncompyle6`
"""

import tarfile


def unpack(filename, destination):
    """
    Extract the contents of a tarfile

    Parameters
    ----------
    filename : Path or str
        Filename to unpack
    destination : Path or str
        Destination to unpack to
    """

    tar = tarfile.open(filename)
    tar.extractall(path=destination)
    tar.close()