#include <Python.h>
static PyObject* value(PyObject* self, PyObject* args) {
    return PyUnicode_FromString("plugin-native-submodule");
}
static PyMethodDef Methods[] = {
    {"value", value, METH_NOARGS, "Return marker."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "native_missing", NULL, -1, Methods
};
PyMODINIT_FUNC PyInit_native_missing(void) { return PyModule_Create(&module); }
