CC:= gcc

CFLAGS:= -fPIC -DDS_VERSION=\"6.4.0\" \
	-I /opt/nvidia/deepstream/deepstream/sources/includes \
	-I /usr/local/cuda/include

LIBS:= -L/opt/nvidia/deepstream/deepstream/lib/ -lnvdsgst_meta -lnvds_meta \
       -L/usr/local/cuda/lib64/ -lcudart \
       -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib/

PKGS:= gstreamer-1.0

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

APP:= deepstream_save_predictions

all: $(APP)

%.o: %.c
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): deepstream_save_predictions.o
	$(CC) -o $(APP) $^ $(LIBS)

clean:
	rm -rf *.o
	rm -rf $(APP)