((python-mode . ((eval . (progn
                           (setq python-coverage--coverage-file-name
                                 (concat (projectile-project-root) "tests/coverage.xml"))
                           (ignore-errors
                             (python-coverage-overlay-mode)))))))
