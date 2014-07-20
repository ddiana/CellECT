yum install -y numpy
yum install -y scipy
yum install -y python-imaging
yum install -y python-matplotlib
yum install -y libsvm
yum install -y python-pip
yum install -y gcc
yum install -y gcc-c++
yum install -y libtiff-devel
sudo yum install sqlite-devel
yum install -y svn
svn checkout http://pylhttps://www.facebook.com/login.php?login_attempt=1ibtiff.googlecode.com/svn/trunk/ pylibtiff-read-only
cd pylibtiff-read-only/
python setup.py build
python setup.py install
cd ..
pip install --user --install-option="--prefix=" -U scikit-learn
yum install -y python-pyside
yum install -y opencv-python
pip install https://pypi.python.org/packages/source/t/termcolor/termcolor-1.1.0.tar.gz
#easy_install openopt
easy_install python-graph-core
easy_install python-graph-dot


