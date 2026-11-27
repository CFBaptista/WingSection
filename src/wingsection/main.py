import tyro


def _wing_section() -> None:
    pass


def main() -> None:
    """Run the WingSection application.

    Examples
    --------
    Run the installed application from a shell:

    $ wingsection
    """
    tyro.cli(_wing_section)


if __name__ == '__main__':
    main()
