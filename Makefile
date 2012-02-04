CXX = g++
LD  = $(CXX)
RM  = rm -f
CXXFLAGS = $(shell pkg-config --cflags opencv) -Wall -O2
LDFLAGS  = $(shell pkg-config --libs opencv)

SOURCE  = tracking.cc
OBJECTS = $(SOURCE:=.o)
TARGET  = tracking

.PHONY: all clean
.SECONDARY:

all: $(TARGET)

clean:
	$(RM) $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

%.cc.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $^

# vim: set noet:
