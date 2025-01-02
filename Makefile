# Personal Makefile variables
#

VERSION='0.9'

FILES=scripts templates_surfaces_32k MacOS Linux WindowsLICENSE

ZIPFILE=T1prep_${VERSION}.zip

# print available commands
help:
	-@echo Available commands:
	-@echo clean zip cp_binaries

# remove .DS_Store files and correct file permissions
clean:
	-@find . -type f -name .DS_Store -exec rm {} \;
	-@chmod -R a+r,g+w,o-w .
	-@find . -type f \( -name "*.sh" \) -exec chmod a+x {} \;

# zip release
zip: clean
	-@echo zip
	-@test ! -d T1prep || rm -r T1prep
	-@mkdir T1prep
	-@cp -rp ${FILES} T1prep
	-@zip ${ZIPFILE} -rm T1prep

# copy binaries after cross-compiling
cp_binaries: 
	-@echo copy binaries
	-@test ! -f ~/Dropbox/GitHub/CAT-Surface/build-*/Progs/*.o || rm ~/Dropbox/GitHub/CAT-Surface/build-*/Progs/*.o
	-@for i in Linux/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-x86_64-pc-linux/Progs/`basename $${i}` Linux/ ; done
	-@for i in Windows/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-i586-mingw32/Progs/`basename $${i}` Windows/ ; done
	-@for i in MacOS/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-native-arm64/Progs/`basename $${i}` MacOS/ ; done
