if (COMMAND "import_remote")
    return()
endif()

set(import_remote_git_bin "git"
        CACHE STRING "Git binary")
set(import_remote_path "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/_imported"
        CACHE STRING "Folder for importing projects")
set(import_remote_bin_path "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/_imported.build"
        CACHE STRING "Binary dir for building projects")
set(import_remote_default_tag "master"
        CACHE STRING "Default git branch for remote projects")
set(import_remote_default_cmake_root "./"
        CACHE STRING "Default subdirectory where to find the main CMakeLists.txt")

find_package(Git REQUIRED)

macro(msg text)
    message(STATUS "{import-remote} ${text}")
endmacro()

# reset at each cmake re-run to consider branch updates
set(data_keys "" CACHE INTERNAL "" FORCE)

function(registry_find key varname)
    string(REGEX MATCH ";${key}#([^;#]+);" match "${data_keys}")
    if ("${CMAKE_MATCH_1}" STREQUAL "")
        set("${varname}" PARENT_SCOPE)
    else()
        set("${varname}" "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endif()
endfunction()

function(registry_add key branch)
    set(data_keys "${data_keys};${key}#${branch};" CACHE INTERNAL "" FORCE)
endfunction()

function(git_init destdir)
    if (EXISTS "${destdir}")
        message(FATAL_ERROR "The folder '${destdir}' already exists. Some manual cleanup may be required.")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" init "${destdir}"
        RESULT_VARIABLE exitstatus ERROR_VARIABLE err
        OUTPUT_QUIET
    )
    if (NOT "${exitstatus}" EQUAL 0)
        message(FATAL_ERROR "failed to initialize empty git repository '${destdir}'\n${err}")
    endif()
endfunction()

function(git_add_remote destdir path)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" remote add "origin" "${path}"
        WORKING_DIRECTORY "${destdir}"
        RESULT_VARIABLE exitstatus ERROR_VARIABLE err
        OUTPUT_QUIET
    )
    if (NOT "${exitstatus}" EQUAL 0)
        message(FATAL_ERROR "failed to add remote to '${path}'\n${err}")
    endif()
endfunction()

function(git_fetch destdir)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" fetch --prune "origin"
        WORKING_DIRECTORY "${destdir}"
        RESULT_VARIABLE exitstatus ERROR_VARIABLE err
        OUTPUT_QUIET
    )
    if (NOT "${exitstatus}" EQUAL 0)
        message(FATAL_ERROR "failed to fetch 'origin' from '${destdir}'\n${err}")
    endif()
endfunction()

function(git_checkout destdir branch)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" checkout "${branch}"
        WORKING_DIRECTORY "${destdir}"
        RESULT_VARIABLE exitstatus ERROR_VARIABLE err
        OUTPUT_QUIET
    )
    if (NOT "${exitstatus}" EQUAL 0)
        message(FATAL_ERROR "failed to checkout branch or tag id '${branch}' from '${destdir}'\n${err}")
    endif()
endfunction()

function(git_submodule_update destdir)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" submodule sync --recursive
        WORKING_DIRECTORY "${destdir}"
        RESULT_VARIABLE exitstatus ERROR_VARIABLE err
        OUTPUT_QUIET
    )
    if (NOT "${exitstatus}" EQUAL 0)
        message(FATAL_ERROR "failed to submodule sync from '${destdir}'\n${err}")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
        WORKING_DIRECTORY "${destdir}"
        RESULT_VARIABLE exitstatus ERROR_VARIABLE err
        OUTPUT_QUIET
    )
    if (NOT "${exitstatus}" EQUAL 0)
        message(FATAL_ERROR "failed to submodule update from '${destdir}'\n${err}")
    endif()
endfunction()

function(git_do_all)
    set(options)
    set(oneValueArgs BRANCH PATH PROJECT_ID VAR KEY)
    set(multiValueArgs)
    cmake_parse_arguments(OPTS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(branch "${OPTS_BRANCH}")
    set(var "${OPTS_VAR}")
    set(project_id "${OPTS_PROJECT_ID}")
    set(key "${OPTS_KEY}")
    string(SUBSTRING "${branch}" 0 1 first_char)
    if (NOT "${first_char}" STREQUAL "/")
        set(destdir "${import_remote_path}/${key}")
        set("${var}" "${destdir}" PARENT_SCOPE)
        if (NOT EXISTS "${destdir}/.git")
            msg("initializing ${project_id}")
            git_init("${destdir}")
            git_add_remote("${destdir}" "${OPTS_PATH}")
        endif()
        msg("updating ${project_id} [origin/${branch}]")
        git_fetch("${destdir}")
        git_checkout("${destdir}" "${branch}")
        git_submodule_update("${destdir}")
    else()
        msg("using ${project_id} from local path '${branch}'")
        set("${var}" "${branch}" PARENT_SCOPE)
        if (NOT EXISTS "${branch}")
            message(FATAL_ERROR "local folder '${branch}' for '${project_id}' is not accessible")
        endif()
    endif()
endfunction()

function (set_var parent_varname value defvalue)
    if ("${value}" STREQUAL "")
        set(value "${defvalue}")
    endif()
    if (NOT "${value}" MATCHES "[a-zA-Z0-9_-]+[.a-zA-Z0-9_ /-]+")
        message(FATAL_ERROR "invalid identifier for ${parent_varname}")
    endif()
    set("${parent_varname}" "${value}" PARENT_SCOPE)
endfunction()

function (make_key parent_varname path)
    set(key "${path}")
    string(REPLACE "/" "--" key "${key}")
    string(REPLACE ":" "-" key "${key}")
    string(REPLACE "--" "-" key "${key}")
    string(REPLACE "--" "-" key "${key}")
    string(REPLACE "--" "-" key "${key}")
    set("${parent_varname}" "${key}" PARENT_SCOPE)
endfunction()

function (import_remote input)
    set(options NO_CMAKE)
    set(oneValueArgs TAG ALIAS SRC)
    set(multiValueArgs)
    cmake_parse_arguments(OPTS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set_var(remote_path "${input}" "")
    set_var(remote_tag "${OPTS_TAG}" "${import_remote_default_tag}")
    set_var(remote_alias "${OPTS_ALIAS}" "${remote_path}")
    make_key(remote_key "${remote_path}")
    registry_find("${remote_key}" "previous_branch")
    if ("${previous_branch}" STREQUAL "")
        # not cloned yet
        registry_add("${remote_key}" "${remote_tag}")
        git_do_all(
                PATH "${remote_path}" BRANCH "${remote_tag}"
                KEY "${remote_key}" PROJECT_ID "${remote_path}"
                VAR dest
        )
        set("${remote_alias}_path" "${dest}" CACHE STRING "" FORCE)
        set(src "${OPTS_SRC}")
        if ("${src}" STREQUAL "")
            set(src "${import_remote_default_cmake_root}")
        endif()
        if (NOT "${OPTS_NO_CMAKE}")
            add_subdirectory("${dest}/${src}"
                    "${import_remote_bin_path}/${remote_key}" EXCLUDE_FROM_ALL)
        endif()
    else()
        if ("${previous_branch}" STREQUAL "${remote_tag}")
            msg("${remote_path}:${remote_tag} already added. Ignoring.")
        else()
            msg("${remote_path}:${remote_tag} already added but with a different branch (from '${previous_branch}'). Ignoring.")
        endif()

    endif()
endfunction()
