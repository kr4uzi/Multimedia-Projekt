# Multimedia-Projekt
MMP WS2015 @ University Augsburg
http://www.multimedia-computing.de/

Author: Markus Kraus

# usage
mmp.exe [config.cfg]

if no command line argument is added, mmp.cfg is used (if existing)

# building
## windows
* download and extract boost, vlfeat and opencv  
* build boost libraries with:
```
bootstrap
b2 address-model=64 --with-system --with-filesystem
```

* edit and adjust the include- and library-paths in:  
boost.props, opencv.props, vlfeat.props and  
matlab.props (if build with matlab support is desired)

## unix
* make sure boost-{filesystem, system}, vlfeat and opencv are installed
* make sure opencv is compiled with libpng12  
or  
* download pngcrush and execute
```
for file in *.png ; do pngcrush "$file" "${file%.png}-crushed.png" && mv "${file%.png}-crushed.png" "$file" ; done
```

* make sure vlfeat's .so/.dylib can be found  
in /svm_light: make  
in /Multimedia Projekt: make ARCH={maci64, glnxa64} (see vlfeat's Makefile for more information)  
e.g. make ARCH=maci64
