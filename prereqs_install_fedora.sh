yum install -y numpy
yum install -y scipy
yum install -y python-imaging
yum install -y python-matplotlib
yum install -y libsvm
yum install -y python-pip
yum install -y gcc
yum install -y gcc-c++
pip install http://downloads.sourceforge.net/project/pyml/PyML-0.7.13.tar.gz?r=&ts=1378622536&use_mirror=softlayer-dal
yum install -y libtiff-devel
yum install -y svn
svn checkout http://pylibtiff.googlecode.com/svn/trunk/ pylibtiff-read-only
cd pylibtiff-read-only/
python setup.py build
python setup.py install
cd ..
yum install -y python-pyside
yum install -y opencv-python
pip install https://pypi.python.org/packages/source/t/termcolor/termcolor-1.1.0.tar.gz
easy_install openopt

