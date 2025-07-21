# Compiler and compiler flags
CC = gcc
# CFLAGS: C compiler flags. -Wall enables all warnings, -g adds debugging info.
CFLAGS = -Wall -g
# LDFLAGS: Linker flags. -lm links the math library.
LDFLAGS = -lm

# The name of the final executable
TARGET = program

# Source files
SRCS = cklearn.c pandac.c plotc.c

OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c pandac.h plotc.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
