#include <jni.h>

extern "C" JNIEXPORT jstring JNICALL
Java_com_smartqrcode_NativeBridge_hello(JNIEnv* env, jobject /*thiz*/) {
    return env->NewStringUTF("native-ok");
}

