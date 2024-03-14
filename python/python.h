#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h>
#include <memory>

namespace py = pybind11;

/// Shorthand notation for defining a data structure
#define OVR_PY_STRUCT(Class, ...) \
    py::class_<Class>(m, #Class, ##__VA_ARGS__)

#define OVR_PY_NAMED_STRUCT(Class, Name, ...) \
    py::class_<Class>(m, Name, ##__VA_ARGS__)

#define OVR_PY_NAMED_INHERITED_STRUCT(Class, Name, Baseclass, ...) \
    py::class_<Class, Baseclass>(m, Name, ##__VA_ARGS__)

/// Shorthand to make a class or struct wrap in shared_ptr instead of the default unique_ptr
#define OVR_PY_STRUCT_PTR(PointsToName, Name, ...) \
    py::class_<PointsToName, std::shared_ptr<PointsToName>>(m, Name, ##__VA_ARGS__)

/// Shorthand notation for defining an enum
#define OVR_PY_ENUM(Name, ...) \
    py::enum_<Name>(m, #Name, ##__VA_ARGS__)

/// Shorthand notation for defining enum members
#define def_value(Class, Value, ...) \
    value(#Value, Class::Value, ##__VA_ARGS__)

/// Shorthand to make a struct/class initializer
#define def_init(...) \
    def(py::init< __VA_ARGS__ >())

/// Shorthand notation for defining most kinds of methods
#define def_class_method(Class, Function, ...) \
    def(#Function, &Class::Function, ##__VA_ARGS__)

#define def_class_method_overload(Class, Function, Name, ReturnType, ...) \
    def(Name, static_cast<ReturnType (Class::*)(__VA_ARGS__)>(&Class::Function))

#define def_class_lambda(Class, Function, Lambda, ...) \
    def(#Function, Lambda, ##__VA_ARGS__)

/// Shorthand notation for defining class/struct fields
#define def_class_field(Class, Field, ...) \
    def_readwrite(#Field, &Class::Field, ##__VA_ARGS__)

#define def_method(Function, ...) \
    m.def(#Function, &Function, ##__VA_ARGS__)

#define def_named_method(Function, Name, ...) \
    m.def(Name, &Function, ##__VA_ARGS__)

#define def_named_method_overload(Function, Name, ReturnType, ...) \
    m.def(Name, static_cast<ReturnType (*)(__VA_ARGS__)>(&Function))

#define def_lambda(Function, Lambda, ...) \
    m.def(#Function, Lambda, ##__VA_ARGS__)
