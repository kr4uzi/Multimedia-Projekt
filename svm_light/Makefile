C++ = g++
C = gcc
DFLAGS =
OFLAGS = -O3
LFLAGS =
CPPFLAGS = -std=c++0x
CFLAGS = -Wall

OBJS = svm.o
COBJS = svm_common.o svm_hideo.o svm_learn.o

libsvm_light: $(COBJS) $(OBJS)
	ar rcs libsvm_light.a $(COBJS) $(OBJS)

all:
	make libsvm_light

clean:
	rm -f $(COBJS) $(OBJS) libsvm_light.a

$(OBJS): %.o: %.cpp
	$(C++) -o $@ $(CPPFLAGS) -c $<

$(COBJS): %.o: %.c
	$(C) -o $@ $(CFLAGS) -c $<
