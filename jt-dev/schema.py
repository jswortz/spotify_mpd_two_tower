import tensorflow as tf

from collections import OrderedDict


class Field:

    def __init__(self, name, mode, kind, transform=None):
        if mode == "REPEATED":
            self.scope = 'sequential'
            kind = getattr(tf, kind[0])
        else:
            self.scope = 'contextual'
            kind = getattr(tf, kind)
        self.name = name
        self.kind = kind
        self.transform = transform

    def to_dense(self, tensor):
        value = '-1' if self.kind == tf.string else -1
        return tf.sparse.to_dense(tensor, default_value=value)

    def to_sparse(self, tensor):
        value = '-1' if self.kind == tf.string else -1
        index = tf.where(tf.not_equal(tensor, value))
        return tf.SparseTensor(index, tf.gather_nd(tensor, index),
                               tf.shape(tensor, out_type=tf.int64))

    def to_feature_column(self, transform):
        if self.transform is None:
            if self.scope == 'sequential':
                function = tf.feature_column.sequence_numeric_column
            else:
                function = tf.feature_column.numeric_column
            return function(self.name)
        assert False


class Schema(OrderedDict):

    def __init__(self, fields):
        fields = list(map(lambda options: Field(**options), fields))
        names = map(lambda field: field.name, fields)
        super().__init__(zip(names, fields))

    def select(self, scope):
        return [name for name, field in self.items() if field.scope == scope]

    def to_feature_spec(self):

        def _process(name):
            if self[name].scope == 'sequential':
                return tf.io.RaggedFeature(self[name].kind)
            else:
                return tf.io.FixedLenFeature([], self[name].kind)

        return {name: _process(name) for name in self.keys()}