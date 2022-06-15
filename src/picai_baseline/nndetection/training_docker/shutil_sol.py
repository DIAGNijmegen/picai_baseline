import os
import shutil


def copyfile(src, dst, **kwargs):
    """Similar to shutil.copyfile but accepts a directory as input for dst"""
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    return shutil.copyfile(src, dst, **kwargs)


def copytree(src, dst, ignore=None):
    """Similar to shutil.copytree but makes sure that copyfile is used for copying"""
    try:
        shutil.copytree(src, dst,
                        ignore=ignore,
                        symlinks=False,
                        ignore_dangling_symlinks=True,
                        copy_function=copyfile)
    except shutil.Error as e:
        non_permission_errors = []
        for error in e.args[0]:
            msg = error[2] if isinstance(error, tuple) else error
            if 'Operation not permitted' not in msg:
                non_permission_errors.append(error)

        if len(non_permission_errors) > 0:
            raise shutil.Error(non_permission_errors)

    return dst
