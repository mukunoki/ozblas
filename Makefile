MAKE        = make #--no-print-directory
include make.inc

ifeq ($(CUDA),yes)
MAKE_DIRS   += src/cuda
endif
ifeq ($(CPU),yes)
MAKE_DIRS   += src/cpu
endif
MAKE_DIRS   += testing

default:
	 @for subdir in $(MAKE_DIRS) ; do \
	 	(cd $$subdir && $(MAKE)) ;\
	done

clean:
	@for subdir in $(MAKE_DIRS) ; do \
		(cd $$subdir && $(MAKE) clean) ;\
	done
