import subprocess
try:
    import setuptools
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    setuptools = use_setuptools()

from setuptools import find_packages, setup  # noqa: F811

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession

links = []
requires = []

try:
    requirements = list(parse_requirements('requirements.txt'))
except Exception:
    # new versions of pip requires a session
    requirements = list(parse_requirements('requirements.txt', session=PipSession()))

for item in requirements:
    link = None
    if getattr(item, 'url', None):   # older pip has url
        link = str(item.url)
    elif getattr(item, 'link', None):  # newer pip has link
        link = str(item.link)

    if link is not None and item.editable:
        command = 'pip install -e {}'.format(link)
        print("Installing editable repo with command: {}".format(command))
        subprocess.run(command.split())
        continue

    elif link is not None:
        links.append(link)

    if item.req:
        requires.append(str(item.req))


# Get version of tensorflow_probability matching installed tensorflow (avoid importing tensorflow because it's slow)
import subprocess
installed_packages = [
    (line.split('==')[0], *line.split('==')[1].split('.')[:2])
    for line in subprocess.run('pip freeze'.split(), stdout=subprocess.PIPE).stdout.decode().split('\n')
    if '==' in line  # avoid github links
]
print(installed_packages)
tf_version = None
for pkg, major_version, minor_version in installed_packages:
    if pkg == 'tensorflow':
        tf_version = (major_version, minor_version)
        break

if tf_version is None:
    raise Exception("Tensorflow is not installed, install tensorflow before installing auto_yolo.")

version_string = '{}.{}'.format(*tf_version)

if int(tf_version[0]) != 1:
    raise Exception("auto_yolo will not work with tensorflow version {}".format(version_string))

version_lookup = {
    "1.15": "0.8",
    "1.14": "0.7",
    "1.13": "0.6",
    "1.12": "0.5",
    "1.11": "0.4",
}

tfp_version = version_lookup[version_string]

requires.append('tensorflow_probability=={}'.format(tfp_version))

setup(
    name='auto_yolo',
    author="Eric Crawford",
    author_email="eric.crawford@mail.mcgill.ca",
    version='0.1',
    packages=find_packages(),
    install_requires=requires,
    dependency_links=links,
)
