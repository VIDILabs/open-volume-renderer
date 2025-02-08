#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OpenXR::openxr_loader" for configuration "MinSizeRel"
set_property(TARGET OpenXR::openxr_loader APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(OpenXR::openxr_loader PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "C;CXX;RC"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/openxr_loader.lib"
  )

list(APPEND _cmake_import_check_targets OpenXR::openxr_loader )
list(APPEND _cmake_import_check_files_for_OpenXR::openxr_loader "${_IMPORT_PREFIX}/lib/openxr_loader.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
