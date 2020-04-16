add_library(Blaze INTERFACE)

target_include_directories(Blaze INTERFACE
        $<BUILD_INTERFACE:${IMPORT_BLAZE_BLAZEROOT}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CURRENT_SOURCE_DIR}/ext/blaze
        GLOBAL
        )

target_compile_features(Blaze INTERFACE cxx_std_17)

find_package(Threads REQUIRED)
target_compile_definitions(Blaze INTERFACE BLAZE_USE_CPP_THREADS)
target_link_libraries(Blaze INTERFACE ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    execute_process(COMMAND wmic cpu get L3CacheSize
            OUTPUT_VARIABLE tmp
            RESULT_VARIABLE flag
            ERROR_QUIET)
    if (flag)
        execute_process(COMMAND wmic cpu get L2CacheSize
                OUTPUT_VARIABLE tmp
                RESULT_VARIABLE flag
                ERROR_QUIET)
    endif (flag)
    if (flag)
        execute_process(COMMAND wmic cpu get L1CacheSize
                OUTPUT_VARIABLE tmp
                RESULT_VARIABLE flag
                ERROR_QUIET)
    endif (flag)
endif (WIN32)

if (UNIX)
    execute_process(COMMAND cat /sys/devices/system/cpu/cpu0/cache/index3/size
            OUTPUT_VARIABLE tmp
            RESULT_VARIABLE flag
            ERROR_QUIET)
    if (flag)
        execute_process(COMMAND cat /sys/devices/system/cpu/cpu0/cache/index2/size
                OUTPUT_VARIABLE tmp
                RESULT_VARIABLE flag
                ERROR_QUIET)
    endif (flag)
    if (flag)
        execute_process(COMMAND cat /sys/devices/system/cpu/cpu0/cache/index1/size
                OUTPUT_VARIABLE tmp
                RESULT_VARIABLE flag
                ERROR_QUIET)
    endif (flag)
endif (UNIX)

if (APPLE)
    execute_process(COMMAND sysctl -n hw.l3cachesize
            OUTPUT_VARIABLE tmp
            RESULT_VARIABLE flag
            ERROR_QUIET)
    if (flag)
        execute_process(COMMAND sysctl -n hw.l2cachesize
                OUTPUT_VARIABLE tmp
                RESULT_VARIABLE flag
                ERROR_QUIET)
    endif (flag)
    if (flag)
        execute_process(COMMAND sysctl -n hw.l1icachesize
                OUTPUT_VARIABLE tmp
                RESULT_VARIABLE flag
                ERROR_QUIET)
    endif (flag)

    if (flag EQUAL 0)
        math(EXPR tmp ${tmp}/1024)  # If successful convert to kibibytes to comply with rest
    endif (flag EQUAL 0)
endif (APPLE)

if (flag)
    message(WARNING "Cache size not found automatically. Using default value as cache size.")
    set(tmp "3072")
endif (flag)

string(REGEX MATCH "([0-9][0-9]+)" tmp ${tmp}) # Get a number containing at least 2 digits in the string tmp
math(EXPR BLAZE_CACHE_SIZE ${tmp}*1024) # Convert to bytes (assuming that the value is given in kibibytes)
set(BLAZE_CACHE_SIZE "${BLAZE_CACHE_SIZE}UL")
target_compile_definitions( Blaze INTERFACE BLAZE_CACHE_SIZE=${BLAZE_CACHE_SIZE} )

target_compile_definitions( Blaze INTERFACE BLAZE_USE_OPTIMIZED_KERNELS=1 )
