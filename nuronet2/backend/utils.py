# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:22:58 2016

@author: Evander
"""

import six

def get_from_module(name, function_list, attribute_name,
                    instantiate=False, **kwargs):
    if(name is None):
        return None
    if(isinstance(name, six.string_types)):
        ret = function_list.get(name)
        if(not ret):
            raise Exception("Invalid {} type {}".format(attribute_name, name))
        if(instantiate and not kwargs):
            return ret()
        elif(instantiate and kwargs):
            return ret(**kwargs)
        elif(not instantiate):
            return ret
    elif(isinstance(name, dict)):
        ret = function_list.get(name.pop('name'))
        if(ret and instantiate):
            return ret(**name)
        elif(ret and not instantiate):
            return ret
        else:
            raise Exception("Invalid parameters passed to {} type" \
                            .format(attribute_name))

