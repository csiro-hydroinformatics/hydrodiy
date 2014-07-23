#
# for the package to work, you need to add the following line
# in ~/.bashrc
# 
#   export PYTHONPATH=$PYTHONPATH:~/.local/lib/python 
#

# Load appropriate module in wafaridev
module load msdm

# files
HYVERS=hydrodiy-0.3
FHOME=~/.local
URL="https://www.dropbox.com/s/d4w7lyzwjodcqyq/hydrodiy-0.3.tar.gz"

# Create folders
cd ~
mkdir tmp
cd tmp

# Remove all files related to package
rm -r $HYVERS*

# Downloading package data
if [ -f $HYVERS.tar.gz ];
then
	rm $HYVERS.tar.gz
fi
echo Downloading $URL
wget $URL -O $HYVERS.tar.gz

# install package
tar -xvf $HYVERS.tar.gz
cd $HYVERS
python setup.py install --home $FHOME

# run tests
find ../$HYVERS -name 'test*py' -exec python '{}' \;

