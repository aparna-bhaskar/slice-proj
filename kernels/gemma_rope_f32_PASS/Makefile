PROJECT = gemma_rope_f32

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

PYTHON_REF = /home/aparna/gemma4-trace/.conda/bin/python

include ../common.mk

data: rope.py
	$(PYTHON_REF) rope.py

kernel.mu.o: data
