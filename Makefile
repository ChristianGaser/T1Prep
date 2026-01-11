.PHONY: help release clean zip cp_binaries
.DEFAULT: help

VERSION="0.2.5"

ZIPFILE=T1Prep_${VERSION}.zip

# print available commands
help:
	-@echo Available commands:
	-@echo clean zip cp_binaries

# remove .DS_Store files and correct file permissions
clean:
	-@find . -type f -name .DS_Store -exec rm {} \;
	-@chmod -R a+r,g+w,o-w .
	-@find . -type f \( -name "*.sh" \) -exec chmod a+x {} \;
	-@find . -type f \( -name "*_ui" \) -exec chmod a+x {} \;

# zip release
zip: release
	-@echo zip
	-@test ! -d T1Prep || rm -r T1Prep
	-@mkdir T1Prep
	-@rsync -av . T1Prep --exclude env --exclude '.*' --exclude Makefile --exclude Windows-Installation.txt --exclude test
	-@zip ${ZIPFILE} -rm T1Prep

# prepare a release
release: clean
	-@sed -i "" "s/version=.*/version=${VERSION}/" scripts/T1Prep
	-@sed -i "" "s/T1PREP_VERSION=.*/T1PREP_VERSION=v${VERSION}/" Dockerfile

# copy binaries after cross-compiling
cp_binaries: 
	-@echo copy binaries
	-@for i in src/t1prep/bin/Linux/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-x86_64-pc-linux/Progs/`basename $${i}` src/t1prep/bin/Linux/ ; done
	-@for i in src/t1prep/bin/Windows/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-x86_64-w64-mingw32/Progs/`basename $${i}` src/t1prep/bin/Windows/ ; done
	-@for i in src/t1prep/bin/MacOS/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-native-arm64/Progs/`basename $${i}` src/t1prep/bin/MacOS/ ; done
	-@for i in src/t1prep/bin/LinuxARM64/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-aarch64-non-elf/Progs/`basename $${i}` src/t1prep/bin/LinuxARM64/ ; done
