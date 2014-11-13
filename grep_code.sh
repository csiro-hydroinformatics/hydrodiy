# Config - bjp code
fsrc='hydrodiy'
ext='py'

# Lookup string
str='scipy'

# Output log file
fout='./grep.log'

echo ------------------------------------------------- > $fout
echo Search for the following string >> $fout
echo [$str] >> $fout
echo in python files within folder >> $fout
echo $fsrc >> $fout
echo ------------------------------------------------- >> $fout
echo >> $fout
echo >> $fout

for file in $(find $fsrc -name '*.'$ext)
	do echo $file
	if grep -nr "$str" $file
	then 
		echo >> $fout
		echo -------------- >> $fout
		echo Found in $file >> $fout
		grep -nr "$str" $file >> $fout
		echo >> $fout
	fi
done
