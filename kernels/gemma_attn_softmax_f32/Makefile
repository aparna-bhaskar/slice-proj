PROJECT = gemma_attn_softmax_f32

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

PYTHON_REF = /home/aparna/gemma4-trace/.conda/bin/python

include ../common.mk

data: softmax.py
	$(PYTHON_REF) softmax.py

kernel.mu.o: data
