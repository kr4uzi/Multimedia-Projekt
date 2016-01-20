VLROOT = ${HOME}/src/vlfeat
C++ = g++
DFLAGS =
OFLAGS = -O3
LFLAGS = -fopenmp -L../svm_light/ -L$(VLROOT)/bin/glnxa64/ -lvl -lsvm_light -lboost_filesystem -lboost_system -lopencv_core -lopencv_highgui -lopencv_imgproc
CFLAGS = -Wall -fopenmp -std=c++0x -I../. -I$(VLROOT) $(shell pkg-config --cflags opencv)

OBJS = annotation.o classification.o config.o evaluation.o helpers.o hog.o image.o inria.o log.o main.o scale_cache.o

all: 
	make mmp

clean:
	rm -f $(OBJS) ./mmp

mmp: $(OBJS)
	$(C++) -o ./mmp $(OBJS) $(LFLAGS)

$(OBJS): %.o: %.cpp
	$(C++) -o $@ $(CFLAGS) -c $<
