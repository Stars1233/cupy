.. _contrib:

Contribution Guide
==================

This is a guide for all contributions to CuPy.
The development of CuPy is running on `the official repository at GitHub <https://github.com/cupy/cupy>`_.
Anyone that wants to register an issue or to send a pull request should read through this document.


Classification of Contributions
-------------------------------

There are several ways to contribute to CuPy community:

1. Registering an issue
2. Sending a pull request (PR)
3. Sending a question to `CuPy's Gitter channel <https://gitter.im/cupy/community>`_, `CuPy User Group <https://groups.google.com/forum/#!forum/cupy>`_, or `StackOverflow <https://stackoverflow.com/questions/tagged/cupy>`_
4. Open-sourcing an external example
5. Writing a post about CuPy

This document mainly focuses on 1 and 2, though other contributions are also appreciated.


Development Cycle
-----------------

This section explains the development process of CuPy.
Before contributing to CuPy, it is strongly recommended to understand the development cycle.

Versioning
~~~~~~~~~~

The versioning of CuPy follows `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_ and a part of `Semantic versioning <https://semver.org/>`_.
The version number consists of three or four parts: ``X.Y.Zw`` where ``X`` denotes the **major version**, ``Y`` denotes the **minor version**, ``Z`` denotes the **revision number**, and the optional ``w`` denotes the prelease suffix.
While the major, minor, and revision numbers follow the rule of semantic versioning, the pre-release suffix follows PEP 440 so that the version string is much friendly with Python eco-system.

**Note that a major update basically does not contain compatibility-breaking changes from the last release candidate (RC).**
This is not a strict rule, though; if there is a critical API bug that we have to fix for the major version, we may add breaking changes to the major version up.

As for the backward compatibility, see :doc:`user_guide/compatibility`.


.. _contrib-release-cycle:

Release Cycle
~~~~~~~~~~~~~

The first one is the track of **stable versions**, which is a series of revision updates for the latest major version.
The second one is the track of **development versions**, which is a series of pre-releases for the upcoming major version.

Consider that ``X.0.0`` is the latest major version and ``Y.0.0``, ``Z.0.0`` are the succeeding major versions.
Then, the timeline of the updates is depicted by the following table.

========== =========== =========== ============
   Date       ver X       ver Y       ver Z
========== =========== =========== ============
  0 weeks    X.0.0rc1    --         --
  4 weeks    X.0.0       Y.0.0a1    --
  8 weeks    X.1.0*      Y.0.0b1    --
 12 weeks    X.2.0*      Y.0.0rc1   --
 16 weeks    --          Y.0.0      Z.0.0a1
========== =========== =========== ============

(* These might be revision releases)

The dates shown in the left-most column are relative to the release of ``X.0.0rc1``.
In particular, each revision/minor release is made four weeks after the previous one of the same major version, and the pre-release of the upcoming major version is made at the same time.
Whether these releases are revision or minor is determined based on the contents of each update.

Note that there are only three stable releases for the versions ``X.x.x``.
During the parallel development of ``Y.0.0`` and ``Z.0.0a1``, the version ``Y`` is treated as an **almost-stable version** and ``Z`` is treated as a development version.

If there is a critical bug found in ``X.x.x`` after stopping the development of version ``X``, we may release a hot-fix for this version at any time.

We create a milestone for each upcoming release at GitHub.
The GitHub milestone is basically used for collecting the issues and PRs resolved in the release.

.. _contrib-git-branches:

Git Branches
~~~~~~~~~~~~

The ``main`` branch is used to develop pre-release versions.
It means that **alpha, beta, and RC updates are developed at the** ``main`` **branch**.
This branch contains the most up-to-date source tree that includes features newly added after the latest major version.

The stable version is developed at the individual branch named as ``vN`` where "N" reflects the version number (we call it a *versioned branch*).
For example, v1.0.0, v1.0.1, and v1.0.2 will be developed at the ``v1`` branch.

**Notes for contributors:**
When you send a pull request, you basically have to send it to the ``main`` branch.
If the change can also be applied to the stable version, a core team member will apply the same change to the stable version so that the change is also included in the next revision update.

If the change is only applicable to the stable version and not to the ``main`` branch, please send it to the versioned branch.
We basically only accept changes to the latest versioned branch (where the stable version is developed) unless the fix is critical.

If you want to make a new feature of the ``main`` branch available in the current stable version, please send a *backport PR* to the stable version (the latest ``vN`` branch).
See the next section for details.

*Note: a change that can be applied to both branches should be sent to the* ``main`` *branch.*
*Each release of the stable version is also merged to the development version so that the change is also reflected to the next major version.*

Feature Backport PRs
~~~~~~~~~~~~~~~~~~~~

We basically do not backport any new features of the development version to the stable versions.
If you desire to include the feature to the current stable version and you can work on the backport work, we welcome such a contribution.
In such a case, you have to send a backport PR to the latest ``vN`` branch.
**Note that we do not accept any feature backport PRs to older versions because we are not running quality assurance workflows (e.g. CI) for older versions so that we cannot ensure that the PR is correctly ported.**

There are some rules on sending a backport PR.

- Start the PR title from the prefix **[backport]**.
- Clarify the original PR number in the PR description (something like "This is a backport of #XXXX").
- (optional) Write to the PR description the motivation of backporting the feature to the stable version.

Please follow these rules when you create a feature backport PR.

Note: PRs that do not include any changes/additions to APIs (e.g. bug fixes, documentation improvements) are usually backported by core dev members.
It is also appreciated to make such a backport PR by any contributors, though, so that the overall development proceeds more smoothly!


Issues and Pull Requests
------------------------

In this section, we explain how to send issues and pull requests (PRs).

AI Policies
~~~~~~~~~~~

In the era of AI explosion, it is exciting to see how AI/LLM tools make programmers' lives easier and boost developers'
productivity, resulting in a paradigm shift that has transformative impact to open source software development. Great
benefits, however, are accompanied by costs that require serious treatment. For example, recently CuPy has received an
enormous amount of bug reports in which AI tools were exploited to search for, e.g., unexpected corner cases of CuPy
which mismatch with NumPy's behaviors but are unlikely hit by human users, or subtleties in CPU/GPU architecture and
execution differences not yet fully understood by AI.

In light of recent incidents, we would like to reiterate that CuPy being a community project is managed and maintained
by humans, not bots or AIs, and human beings have limited bandwidth and capacity in triaging and responding to bug
reports in addition to other development activities. A flooded issue tracker or PR review queue also hinders the
discoverability of useful information and discussions.

To ensure a smooth transition to AI-assisted development, we kindly encourage users to follow these guidelines:

- Please do not spam CuPy's issue tracker with AI-based contents.
- Please do not misuse the CuPy community as a testbed for your AI development or experiment.
- Please understand :doc:`certain deviations from NumPy <user_guide/difference>` are expected.
- Please only file bug reports that actually impact daily/production use of CuPy, not for hypothetical/pedantic
  scenarios.

The CuPy team reserves rights to close any bug reports proactively and preemptively without further notices, if
it is determined by the team that the above guidelines are not followed by the reporter.

If you believe your bug reports are legitimate and can be encountered in real life, but get closed by accident,
please kindly let us know by pinging ``@cupy/triage-team`` in the closed issue.

How to File an Issue
~~~~~~~~~~~~~~~~~~~~

(Work in progress.)

How to Send a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you can write code to fix an issue, we encourage to send a PR.

First of all, before starting to write any code, do not forget to confirm the following points.

- Read through the :ref:`coding-guide` and :ref:`testing-guide`.
- Check the appropriate branch that you should send the PR following :ref:`contrib-git-branches`.
  If you do not have any idea about selecting a branch, please choose the ``main`` branch.

In particular, **check the branch before writing any code.**
The current source tree of the chosen branch is the starting point of your change.

After writing your code **(including unit tests and hopefully documentations!)**, send a PR on GitHub.
You have to write a precise explanation of **what** and **how** you fix;
it is the first documentation of your code that developers read, which is a very important part of your PR.

Once you send a PR, it is automatically tested on ``GitHub Actions``.
After the automatic test passes, core developers will start reviewing your code.
Note that this automatic PR test only includes CPU tests.

.. note::

   We are also running continuous integration with GPU tests for the ``main`` branch and the versioned branch of the latest major version.
   Since this service is currently running on our internal server, we do not use it for automatic PR tests to keep the server secure.

If you are planning to add a new feature or modify existing APIs, **it is recommended to open an issue and discuss the design first.**
The design discussion needs lower cost for the core developers than code review.
Following the consequences of the discussions, you can send a PR that is smoothly reviewed in a shorter time.

Even if your code is not complete, you can send a pull request as a *work-in-progress PR* by putting the ``[WIP]`` prefix to the PR title.
If you write a precise explanation about the PR, core developers and other contributors can join the discussion about how to proceed the PR.
WIP PR is also useful to have discussions based on a concrete code.


.. _coding-guide:

Coding Guidelines
-----------------

.. note::

   Coding guidelines are updated at v5.0.
   Those who have contributed to older versions should read the guidelines again.

We use `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ and a part of `OpenStack Style Guidelines <https://docs.openstack.org/developer/hacking/>`_ related to general coding style as our basic style guidelines.

You can use ``pre-commit`` to check your code.
First install it with the following command::

  $ pip install pre-commit

And check your code with::

  $ pre-commit run -a

The above command runs various checks listed in the ``.pre-commit-config.yaml`` file, including linting (detect potential errors), formatting (enforce PEP8), and so on.
If you want to automate the checking process, you can install the pre-commit hook::

  $ pre-commit install

Once installed, your code should be checked each time you commit.
Before sending a pull request, be sure to check that your code passes the ``pre-commit`` checking.

Note that ``pre-commit`` check is not perfect.
It does not check some of the style guidelines.
Here is a (not-complete) list of the rules that ``pre-commit`` cannot check.

* Relative imports are prohibited. [H304]
* Importing non-module symbols is prohibited.
* Import statements must be organized into three parts: standard libraries, third-party libraries, and internal imports. [H306]

In addition, we restrict the usage of *shortcut symbols* in our code base.
They are symbols imported by packages and sub-packages of ``cupy``.
For example, ``cupy.cuda.Device`` is a shortcut of ``cupy.cuda.device.Device``.
**It is not allowed to use such shortcuts in the ``cupy`` library implementation**.
Note that you can still use them in :tree:`tests` and :tree:`examples` directories.

Once you send a pull request, your coding style is automatically checked by `GitHub Actions`.
The reviewing process starts after the check passes.

The CuPy is designed based on NumPy's API design. CuPy's source code and documents contain the original NumPy ones.
Please note the following when writing the document.

* In order to identify overlapping parts, it is preferable to add some remarks
  that this document is just copied or altered from the original one. It is
  also preferable to briefly explain the specification of the function in a
  short paragraph, and refer to the corresponding function in NumPy so that
  users can read the detailed document. However, it is possible to include a
  complete copy of the document with such a remark if users cannot summarize
  in such a way.
* If a function in CuPy only implements a limited amount of features in the
  original one, users should explicitly describe only what is implemented in
  the document.

For changes that modify or add new Cython files, please make sure the pointer types follow these guidelines (`#1913 <https://github.com/cupy/cupy/issues/1913>`_).

* Pointers should be ``void*`` if only used within Cython, or ``intptr_t`` if exposed to the Python space.
* Memory sizes should be ``size_t``.
* Memory offsets should be ``ptrdiff_t``.

.. note::

     We are incrementally enforcing the above rules, so some existing code may not follow the above guidelines, but please ensure all new contributions do.

.. _testing-guide:

Unit Testing
------------

Testing is one of the most important part of your code.
You must write test cases and verify your implementation by following our testing guide.

Note that we are using pytest and mock package for testing, so install them before writing your code::

  $ pip install pytest mock

How to Run Tests
~~~~~~~~~~~~~~~~

In order to run unit tests at the repository root, you first have to build Cython files in place by running the following command::

  $ pip install --no-build-isolation -e .

.. note::

  When you modify ``*.pxd`` files, before running ``pip install --no-build-isolation -e .``, you must clean ``*.cpp`` and ``*.so`` files once with the following command, because Cython does not automatically rebuild those files nicely::

    $ git clean -fdx

  In addition, you need to manually install the build-time dependencies listed in the ``build-system.requires`` section of ``pyproject.toml`` before running the above command::

    $ pip install "setuptools>=77" wheel "Cython>=3,<3.2" "fastrlock>=0.5"

Once Cython modules are built, you can run unit tests by running the following command at the repository root::

  $ python -m pytest

CUDA must be installed to run unit tests.

Some GPU tests require cuDNN to run.
In order to skip unit tests that require cuDNN, specify ``-m='not cudnn'`` option::

  $ python -m pytest path/to/your/test.py -m='not cudnn'

Some GPU tests involve multiple GPUs.
If you want to run GPU tests with insufficient number of GPUs, specify the number of available GPUs to ``CUPY_TEST_GPU_LIMIT``.
For example, if you have only one GPU, launch ``pytest`` by the following command to skip multi-GPU tests::

  $ export CUPY_TEST_GPU_LIMIT=1
  $ python -m pytest path/to/gpu/test.py

Following this naming convention, you can run all the tests by running the following command at the repository root::

  $ python -m pytest

Or you can also specify a root directory to search test scripts from::

  $ python -m pytest tests/cupy_tests     # to just run tests of CuPy
  $ python -m pytest tests/install_tests  # to just run tests of installation modules

If you modify the code related to existing unit tests, you must run appropriate commands.

Test File and Directory Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests are put into the :tree:`tests/cupy_tests` directory.
In order to enable test runner to find test scripts correctly, we are using special naming convention for the test subdirectories and the test scripts.

* The name of each subdirectory of ``tests`` must end with the ``_tests`` suffix.
* The name of each test script must start with the ``test_`` prefix.

When we write a test for a module, we use the appropriate path and file name for the test script whose correspondence to the tested module is clear.
For example, if you want to write a test for a module ``cupy.x.y.z``, the test script must be located at ``tests/cupy_tests/x_tests/y_tests/test_z.py``.

How to Write Tests
~~~~~~~~~~~~~~~~~~

There are many examples of unit tests under the :tree:`tests` directory, so reading some of them is a good and recommended way to learn how to write tests for CuPy.
They simply use the :mod:`unittest` package of the standard library, while some tests are using utilities from :mod:`cupy.testing`.

In addition to the :ref:`coding-guide` mentioned above, the following rules are applied to the test code:

* All test classes must inherit from :class:`unittest.TestCase`.
* Use :mod:`unittest` features to write tests, except for the following cases:

    * Use ``assert`` statement instead of ``self.assert*`` methods (e.g., write ``assert x == 1`` instead of ``self.assertEqual(x, 1)``).
    * Use ``with pytest.raises(...):`` instead of ``with self.assertRaises(...):``.

.. note::

   We are incrementally applying the above style.
   Some existing tests may be using the old style (``self.assertRaises``, etc.), but all newly written tests should follow the above style.

In order to write tests for multiple GPUs, use ``cupy.testing.multi_gpu()`` decorators instead::

  import unittest
  from cupy import testing

  class TestMyFunc(unittest.TestCase):
      ...

      @testing.multi_gpu(2)  # specify the number of required GPUs here
      def test_my_two_gpu_func(self):
          ...

If your test requires too much time, add ``cupy.testing.slow`` decorator.
The test functions decorated by ``slow`` are skipped if ``-m='not slow'`` is given::

  import unittest
  from cupy import testing

  class TestMyFunc(unittest.TestCase):
      ...

      @testing.slow
      def test_my_slow_func(self):
          ...

Once you send a pull request, GitHub Actions automatically checks if your code meets our coding guidelines described above.
Since GitHub Actions does not support CUDA, we cannot run unit tests automatically.
The reviewing process starts after the automatic check passes.
Note that reviewers will test your code without the option to check CUDA-related code.

.. note::
   Some of numerically unstable tests might cause errors irrelevant to your changes.
   In such a case, we ignore the failures and go on to the review process, so do not worry about it!


Documentation
-------------

When adding a new feature to the framework, you also need to document it in the reference.

.. note::

   If you are unsure about how to fix the documentation, you can submit a pull request without doing so.
   Reviewers will help you fix the documentation appropriately.

The documentation source is stored under `docs directory <https://github.com/cupy/cupy/tree/main/docs>`_ and written in `reStructuredText <http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ format.

To build the documentation, you need to install `Sphinx <http://www.sphinx-doc.org/>`_::

  $ pip install -r docs/requirements.txt

Then you can build the documentation in HTML format locally::

  $ cd docs
  $ make html

HTML files are generated under ``build/html`` directory.
Open ``index.html`` with the browser and see if it is rendered as expected.

.. note::

   Docstrings (documentation comments in the source code) are collected from the installed CuPy module.
   If you modified docstrings, make sure to install the module (e.g., using `pip install --no-build-isolation -e .`) before building the documentation.


Tips for Developers
-------------------

Here are some tips for developers hacking CuPy source code.

Install as Editable
~~~~~~~~~~~~~~~~~~~

During the development we recommend using ``pip`` with ``-e`` option to install as editable mode::

  $ pip install --no-build-isolation -e .

The ``--no-build-isolation`` option enables incremental compilation.
More specifically, the build runs inside the ``build/temp.*`` directory instead of the isolated temporary directory created for each invocation, reusing object files generated in the previous build.

Please note that even with ``-e``, you will have to rerun ``pip install --no-build-isolation -e .`` to regenerate C++ sources using Cython if you modified Cython source files (e.g., ``*.pyx`` files).

Use ccache
~~~~~~~~~~

``NVCC`` environment variable can be specified at the build time to use the custom command instead of ``nvcc`` .
You can speed up the rebuild using `ccache <https://ccache.dev/>`_ (v3.4 or later) by::

  $ export NVCC='ccache nvcc'

Limit Architecture
~~~~~~~~~~~~~~~~~~

Use ``CUPY_NVCC_GENERATE_CODE`` environment variable to reduce the build time by limiting the target CUDA architectures.
For example, if you only run your CuPy build with NVIDIA P100 and V100, you can use::

  $ export CUPY_NVCC_GENERATE_CODE=arch=compute_60,code=sm_60;arch=compute_70,code=sm_70

See :doc:`reference/environment` for the description.

Development on Microsoft Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy uses symbolic links in the source tree.
If you are developing on Windows (outside of WSL), you will need additional setup before cloning the CuPy repository.

* Run ``git config --global core.symlinks true`` to enable symbolic link support in Git.
* `Activate Developer Mode <https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development#activate-developer-mode>`_ to allow Git using symbolic link without Administrator privilege.

Once configured, you can clone the repository by ``git clone --recursive https://github.com/cupy/cupy.git``.
