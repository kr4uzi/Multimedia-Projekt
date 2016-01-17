# Multimedia-Projekt
MMP WS2015 @ University Augsburg
http://www.multimedia-computing.de/

Author: Markus Kraus

usage: 
mmp.exe [config.cfg]

if no command line argument is added, mmp.cfg is used (if existing)

building:
boost libraries with: 
bootstrap
b2 address-model=64 --with-system --with-filesystem

edit and adjust the include- and library-paths in: 
boost.props, opencv.props, vlfeat.props and 
matlab.props (if build with matlab support is desired)
