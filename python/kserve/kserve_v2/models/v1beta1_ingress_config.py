# Copyright 2022 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8

"""
    KServe

    Python SDK for KServe  # noqa: E501

    The version of the OpenAPI document: v0.1
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six

from kserve.configuration import Configuration


class V1beta1IngressConfig(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'domain_template': 'str',
        'ingress_class_name': 'str',
        'ingress_domain': 'str',
        'ingress_gateway': 'str',
        'ingress_service': 'str',
        'local_gateway': 'str',
        'local_gateway_service': 'str'
    }

    attribute_map = {
        'domain_template': 'domainTemplate',
        'ingress_class_name': 'ingressClassName',
        'ingress_domain': 'ingressDomain',
        'ingress_gateway': 'ingressGateway',
        'ingress_service': 'ingressService',
        'local_gateway': 'localGateway',
        'local_gateway_service': 'localGatewayService'
    }

    def __init__(self, domain_template=None, ingress_class_name=None, ingress_domain=None, ingress_gateway=None, ingress_service=None, local_gateway=None, local_gateway_service=None, local_vars_configuration=None):  # noqa: E501
        """V1beta1IngressConfig - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._domain_template = None
        self._ingress_class_name = None
        self._ingress_domain = None
        self._ingress_gateway = None
        self._ingress_service = None
        self._local_gateway = None
        self._local_gateway_service = None
        self.discriminator = None

        if domain_template is not None:
            self.domain_template = domain_template
        if ingress_class_name is not None:
            self.ingress_class_name = ingress_class_name
        if ingress_domain is not None:
            self.ingress_domain = ingress_domain
        if ingress_gateway is not None:
            self.ingress_gateway = ingress_gateway
        if ingress_service is not None:
            self.ingress_service = ingress_service
        if local_gateway is not None:
            self.local_gateway = local_gateway
        if local_gateway_service is not None:
            self.local_gateway_service = local_gateway_service

    @property
    def domain_template(self):
        """Gets the domain_template of this V1beta1IngressConfig.  # noqa: E501


        :return: The domain_template of this V1beta1IngressConfig.  # noqa: E501
        :rtype: str
        """
        return self._domain_template

    @domain_template.setter
    def domain_template(self, domain_template):
        """Sets the domain_template of this V1beta1IngressConfig.


        :param domain_template: The domain_template of this V1beta1IngressConfig.  # noqa: E501
        :type: str
        """

        self._domain_template = domain_template

    @property
    def ingress_class_name(self):
        """Gets the ingress_class_name of this V1beta1IngressConfig.  # noqa: E501


        :return: The ingress_class_name of this V1beta1IngressConfig.  # noqa: E501
        :rtype: str
        """
        return self._ingress_class_name

    @ingress_class_name.setter
    def ingress_class_name(self, ingress_class_name):
        """Sets the ingress_class_name of this V1beta1IngressConfig.


        :param ingress_class_name: The ingress_class_name of this V1beta1IngressConfig.  # noqa: E501
        :type: str
        """

        self._ingress_class_name = ingress_class_name

    @property
    def ingress_domain(self):
        """Gets the ingress_domain of this V1beta1IngressConfig.  # noqa: E501


        :return: The ingress_domain of this V1beta1IngressConfig.  # noqa: E501
        :rtype: str
        """
        return self._ingress_domain

    @ingress_domain.setter
    def ingress_domain(self, ingress_domain):
        """Sets the ingress_domain of this V1beta1IngressConfig.


        :param ingress_domain: The ingress_domain of this V1beta1IngressConfig.  # noqa: E501
        :type: str
        """

        self._ingress_domain = ingress_domain

    @property
    def ingress_gateway(self):
        """Gets the ingress_gateway of this V1beta1IngressConfig.  # noqa: E501


        :return: The ingress_gateway of this V1beta1IngressConfig.  # noqa: E501
        :rtype: str
        """
        return self._ingress_gateway

    @ingress_gateway.setter
    def ingress_gateway(self, ingress_gateway):
        """Sets the ingress_gateway of this V1beta1IngressConfig.


        :param ingress_gateway: The ingress_gateway of this V1beta1IngressConfig.  # noqa: E501
        :type: str
        """

        self._ingress_gateway = ingress_gateway

    @property
    def ingress_service(self):
        """Gets the ingress_service of this V1beta1IngressConfig.  # noqa: E501


        :return: The ingress_service of this V1beta1IngressConfig.  # noqa: E501
        :rtype: str
        """
        return self._ingress_service

    @ingress_service.setter
    def ingress_service(self, ingress_service):
        """Sets the ingress_service of this V1beta1IngressConfig.


        :param ingress_service: The ingress_service of this V1beta1IngressConfig.  # noqa: E501
        :type: str
        """

        self._ingress_service = ingress_service

    @property
    def local_gateway(self):
        """Gets the local_gateway of this V1beta1IngressConfig.  # noqa: E501


        :return: The local_gateway of this V1beta1IngressConfig.  # noqa: E501
        :rtype: str
        """
        return self._local_gateway

    @local_gateway.setter
    def local_gateway(self, local_gateway):
        """Sets the local_gateway of this V1beta1IngressConfig.


        :param local_gateway: The local_gateway of this V1beta1IngressConfig.  # noqa: E501
        :type: str
        """

        self._local_gateway = local_gateway

    @property
    def local_gateway_service(self):
        """Gets the local_gateway_service of this V1beta1IngressConfig.  # noqa: E501


        :return: The local_gateway_service of this V1beta1IngressConfig.  # noqa: E501
        :rtype: str
        """
        return self._local_gateway_service

    @local_gateway_service.setter
    def local_gateway_service(self, local_gateway_service):
        """Sets the local_gateway_service of this V1beta1IngressConfig.


        :param local_gateway_service: The local_gateway_service of this V1beta1IngressConfig.  # noqa: E501
        :type: str
        """

        self._local_gateway_service = local_gateway_service

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, V1beta1IngressConfig):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, V1beta1IngressConfig):
            return True

        return self.to_dict() != other.to_dict()