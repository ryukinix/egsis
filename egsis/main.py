import typer

cli = typer.Typer(help="EGSIS CLI")


@cli.command()
def hello_world():
    print("Hello world!")
