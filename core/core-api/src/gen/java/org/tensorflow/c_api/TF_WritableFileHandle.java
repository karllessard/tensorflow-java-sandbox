// Targeted by JavaCPP version 1.5.2-SNAPSHOT: DO NOT EDIT THIS FILE

package org.tensorflow.c_api;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.tensorflow.c_api.global.tensorflow.*;

// #endif

@Opaque @Properties(inherit = org.tensorflow.c_api.presets.tensorflow.class)
public class TF_WritableFileHandle extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TF_WritableFileHandle() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TF_WritableFileHandle(Pointer p) { super(p); }
}