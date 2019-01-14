#SPACE
NULLSTRING :=
SPACE := $(NULLSTRING)

BASEDIR := $(abspath .)
SRCDIR := $(BASEDIR)/src
HEADERDIR := $(SRCDIR)/headers
CSRCDIR := $(SRCDIR)/csrc

BUILDDIR := $(BASEDIR)/build
OUTPUTDIR := $(BASEDIR)/output

CC := riscv64-unknown-elf-gcc
CXX := riscv64-unknown-elf-g++ 
CFLAGS += -lm
default: all

# Make Flags
TEST ?= 1
TRAIN ?= 0

ifeq ($(TEST),1)
	CONFIGTEST := _test
endif

ifeq ($(TRAIN),1)
	CONFIGTRAIN := _train
endif

SAVETRANDATA ?= 0 # 0-Not save & 1-save

PREDEFINE += SAVETRANDATA=$(SAVETRANDATA) \
             TEST=$(TEST) \
			       TRAIN=$(TRAIN) \

PREDEF := $(addprefix -D$(SPACE),$(PREDEFINE))
CFLAGS += $(PREDEF)

all: cnn$(CONFIGTRAIN)$(CONFIGTEST)
cnn$(CONFIGTRAIN)$(CONFIGTEST): $(CSRCDIR)/*.c
	@echo "[C] Make CNN $(CONFIGTRAIN) $(CONFIGTEST) Module"
	mkdir -p $(BUILDDIR)
	$(CC) $^ -I$(HEADERDIR) $(CFLAGS) -o $(BUILDDIR)/$@

.PHONY: run
run:
	$(BUILDDIR)/cnn$(CONFIGTRAIN)$(CONFIGTEST)

.PHONY: clean
clean:
	rm -f $(BUILDDIR)/cnn_*
	rmdir $(BUILDDIR)

.PHONY: dist-clean
dist-clean: clean
	rm -rf $(OUTPUTDIR)

