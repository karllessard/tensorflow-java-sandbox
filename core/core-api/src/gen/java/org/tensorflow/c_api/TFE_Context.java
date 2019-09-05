// Targeted by JavaCPP version 1.5.2-SNAPSHOT: DO NOT EDIT THIS FILE

package org.tensorflow.c_api;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.tensorflow.c_api.global.tensorflow.*;


// "Context" under which operations/functions are executed. It encapsulates
// things like the available devices, resource manager etc.
// TFE_Context must outlive all tensor handles created using it. In other
// words, TFE_DeleteContext() must be called after all tensor handles have
// been deleted (with TFE_DeleteTensorHandle).
//
// TODO(ashankar): Merge with TF_Session?
@Opaque @Properties(inherit = org.tensorflow.c_api.presets.tensorflow.class)
public class TFE_Context extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TFE_Context() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TFE_Context(Pointer p) { super(p); }
}
