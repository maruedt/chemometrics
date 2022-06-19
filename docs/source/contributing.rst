Contributor's guide
===================

Thank you for your interest in contributing to chemometrics. Your help is
greatly appreciated. The source code is hosted at
https://github.com/maruedt/chemometrics. You can obtain the latest changes to
chemometrics by cloning the repository.

.. note::

  If you have any questions, would like to request features, discovered a bug
  or have a feature request, please feel free to reach out to me or open an
  issue on Github.

General points
--------------

Some general points to consider when working on chemometrics:

* chemometrics extends the capabilities of scikit-learn for chemometric
  applications. It should be easy to include chemometrics objects into
  existing scikit-learn workflows (e.g. pipelines).
* Build on existing functionality of scikit-learn, numpy and scipy (e.g.
  consider subclassing the existing scikit-learn classes, if they provide the
  core functionality you would like to implement).
* Code coverage with unittests should be >=90%, ideally 100% coverage.
* Your code should come with documentation of similar or better quality than
  the rest of chemometrics.


Focus areas
-----------

Some topics which need improvement:

* Implement readers for commercial spectroscopic and other chemical data
  formats.
* Integrate existing 3rd party packages into chemometrics. Generally, there
  are some very useful small packages already available for
  chemometrics in Python. However, those packages are not curated under a
  common framework which reduces reusability and makes it more difficult for
  users to use those methods. Integrating the functionality into
  chemometrics simplifies the usage.
* Add example sections in docstrings.




Licensing
---------

Code pushed to chemometrics will be released under GPLv3. If you are pushing to
chemometrics, you agree to the provided code under this license. 
