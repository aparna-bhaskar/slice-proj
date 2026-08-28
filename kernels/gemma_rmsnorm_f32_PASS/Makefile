PROJECT = gemma_rmsnorm_f32

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

PYTHON_REF = /home/aparna/gemma4-trace/.conda/bin/python

include ../common.mk

data: rmsnorm.py
	$(PYTHON_REF) rmsnorm.py

kernel.mu.o: MU_CFLAGS += -fno-math-errno
kernel.mu.o: data
