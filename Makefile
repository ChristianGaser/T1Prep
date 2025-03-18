.PHONY: help docker clean zip cp_binaries
.DEFAULT: help

VERSION='0.9'

FILES=scripts src data bin LICENSE README.md requirements.txt

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

docker:
	docker build --rm -t ChristianGaser/T1Prep:$(tag) \
	--build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
	--build-arg VCS_REF=`git rev-parse --short HEAD` \
	--build-arg VERSION=`python3 setup.py --version` .

# zip release
zip: clean
	-@echo zip
	-@test ! -d T1Prep || rm -r T1Prep
	-@mkdir T1Prep
	-@cp -rp ${FILES} T1Prep
	-@zip ${ZIPFILE} -rm T1Prep

# copy binaries after cross-compiling
cp_binaries: 
	-@echo copy binaries
	-@for i in bin/Linux/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-x86_64-pc-linux/Progs/`basename $${i}` bin/Linux/ ; done
	-@for i in bin/Windows/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-x86_64-w64-mingw32/Progs/`basename $${i}` bin/Windows/ ; done
	-@for i in bin/MacOS/CAT*; do cp ~/Dropbox/GitHub/CAT-Surface/build-native-arm64/Progs/`basename $${i}` bin/MacOS/ ; done
