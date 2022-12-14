CXX = g++
OPENCV = $$(pkg-config --cflags --libs opencv)
EXE = main.out
SRC = main.cpp

.PHONY: clean

all: main.o hog.o
	$(CXX) -o main.out $(OPENCV) main.o hog.o
main.o: main.cpp hog.hpp
	$(CXX) -c main.cpp
hog.o: hog.cpp hog.hpp
	$(CXX) -c hog.cpp

clean: 
	rm -f *.o *.out
	rm -f images/*result*
