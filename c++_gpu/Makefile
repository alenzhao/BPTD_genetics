CC=	g++
#OPT=	-std=c++11 -static-libstdc++ -Wall -pg
OPT=	-std=c++11 -static-libstdc++
LIBS=	-lm -lpthread
SRCS=	main.cpp
OBJS=	main.o

EXECUTABLE=	main



all: main clean mv


$(OBJS): $(SRCS)
	$(CC) -c $*.cpp $(OPT)

main: $(OBJS)
	$(CC) -o $(EXECUTABLE) $(OBJS) $(OPT) $(LIBS)

clean:
	-rm -f *.o

mv:
	@chmod 755 $(EXECUTABLE)

