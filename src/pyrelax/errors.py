class RelAlgError(Exception):
    """Base class for all errors raised by this package."""


class RelAlgSyntaxError(RelAlgError):
    """The query text could not be parsed."""


class RelAlgExecutionError(RelAlgError):
    """The query parsed fine but could not be executed (unknown relation,
    unknown/ambiguous column, degree mismatch, unsupported construct...)."""
