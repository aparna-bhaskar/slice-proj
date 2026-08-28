PROJECT = gemma_gelu_f32

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

PYTHON_REF = /home/aparna/gemma4-trace/.conda/bin/python

include ../common.mk

data: gelu.py
	$(PYTHON_REF) gelu.py

kernel.mu.o: data
