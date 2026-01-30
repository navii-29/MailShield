from setuptools import setup,find_packages

HYPHEN_E_DoT = '-e .'

def get_requirements(file_path: str):
    requirements = []

    with open(file_path,encoding='utf-8') as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip and line.strip() != HYPHEN_E_DoT
        ]

        return requirements


setup(
    name = 'Spam_detection',
    version = '0.0.1',
    author='neetcoder_29',
    author_email = 'valoroustruth@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt"),
)