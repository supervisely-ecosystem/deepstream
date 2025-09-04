#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

static FILE *predictions_file = NULL;
static guint frame_number = 0;

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        
        // Start JSON object for this frame
        fprintf(predictions_file, "{\"frame_id\":%u,\"timestamp\":%lu,\"objects\":[", 
                frame_number, (unsigned long)frame_meta->ntp_timestamp);
        
        gboolean first_object = TRUE;
        
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            
            if (!first_object) {
                fprintf(predictions_file, ",");
            }
            first_object = FALSE;
            
            // Write object data as JSON
            fprintf(predictions_file, 
                "{\"bbox\":{\"left\":%.2f,\"top\":%.2f,\"width\":%.2f,\"height\":%.2f},"
                "\"confidence\":%.3f,\"class_id\":%u,\"track_id\":%lu,\"class_name\":\"%s\"}",
                obj_meta->rect_params.left,
                obj_meta->rect_params.top, 
                obj_meta->rect_params.width,
                obj_meta->rect_params.height,
                obj_meta->confidence,
                obj_meta->class_id,
                obj_meta->object_id,
                obj_meta->obj_label ? obj_meta->obj_label : "unknown");
                
            num_rects++;
        }
        
        // Close JSON object for this frame
        fprintf(predictions_file, "]}\n");
        fflush(predictions_file);
    }
    
    frame_number++;
    g_print ("Frame Number = %d Number of objects = %d\n", frame_number, num_rects);
    
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      if (predictions_file) {
        fclose(predictions_file);
        predictions_file = NULL;
      }
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      if (predictions_file) {
        fclose(predictions_file);
        predictions_file = NULL;
      }
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstElement *converter1 = (GstElement *) data;
  
  GstPad *converter_sink_pad = gst_element_get_static_pad (converter1, "sink");
  
  if (gst_pad_link (decoder_src_pad, converter_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link decoder src pad to converter sink pad\n");
  } else {
    g_print ("Linked decoder src pad to converter sink pad successfully\n");
  }
  
  gst_object_unref (converter_sink_pad);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *decodebin = NULL,
      *converter1 = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL, *tracker = NULL;
  GstElement *transform = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;

  if (argc != 3) {
    g_printerr ("Usage: %s <input_file> <predictions_file>\n", argv[0]);
    return -1;
  }

  // Open predictions file
  predictions_file = fopen(argv[2], "w");
  if (!predictions_file) {
    g_printerr ("Failed to open predictions file: %s\n", argv[2]);
    return -1;
  }

  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  // Create pipeline elements
  pipeline = gst_pipeline_new ("dstest1-pipeline");
  source = gst_element_factory_make ("filesrc", "file-source");
  decodebin = gst_element_factory_make ("decodebin", "decoder-bin");
  converter1 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter-1");
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  tracker = gst_element_factory_make ("nvtracker", "tracker");
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  if (!pipeline || !source || !decodebin || !converter1 || !streammux || !pgie
      || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  // Set properties
  g_object_set (G_OBJECT (source), "location", argv[1], NULL);
  g_object_set (G_OBJECT (streammux), "width", 640, "height", 640,
      "batch-size", 1,
      "batched-push-timeout", 4000000, NULL);
  g_object_set (G_OBJECT (pgie), "config-file-path", "config_infer_dfine.txt", NULL);
  g_object_set (G_OBJECT (tracker), "ll-lib-file", 
      "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
      "ll-config-file", "config_tracker_NvDCF_perf.yml", NULL);

  // Add message handler
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  // Add elements to pipeline  
  gst_bin_add_many (GST_BIN (pipeline),
      source, decodebin, converter1, streammux, pgie, tracker, nvvidconv, nvosd, transform, sink, NULL);

  // Set up dynamic pad callback for decodebin
  g_signal_connect (decodebin, "pad-added", G_CALLBACK (cb_newpad), converter1);

  // Link static elements
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";

  sinkpad = gst_element_request_pad_simple (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (converter1, "src");
  if (!srcpad) {
    g_printerr ("Converter1 request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link converter1 to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  // Link source to decodebin
  if (!gst_element_link (source, decodebin)) {
    g_printerr ("Elements could not be linked: source to decodebin. Exiting.\n");
    return -1;
  }

  // Link post-streammux elements
  if (!gst_element_link_many (streammux, pgie, tracker, nvvidconv, nvosd, transform, sink, NULL)) {
    g_printerr ("Elements could not be linked: post-streammux. Exiting.\n");
    return -1;
  }

  // Add probe to get metadata
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  // Start playing
  g_print ("Running...\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  // Run loop
  g_main_loop_run (loop);

  // Clean up
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);

  if (predictions_file) {
    fclose(predictions_file);
  }

  return 0;
}