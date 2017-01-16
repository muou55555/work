# Copyright (C) 2013  Google Inc.
#
# This file is part of YouCompleteMe.
#
# YouCompleteMe is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# YouCompleteMe is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with YouCompleteMe.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *  # noqa

import requests
import urllib.parse
import json
from future.utils import native
from base64 import b64decode, b64encode
from retries import retries
from requests_futures.sessions import FuturesSession
from ycm.unsafe_thread_pool_executor import UnsafeThreadPoolExecutor
from ycm import vimsupport
from ycmd.utils import ToBytes
from ycmd.hmac_utils import CreateRequestHmac, CreateHmac, SecureBytesEqual
from ycmd.responses import ServerError, UnknownExtraConf

_HEADERS = {'content-type': 'application/json'}
_EXECUTOR = UnsafeThreadPoolExecutor( max_workers = 30 )
# Setting this to None seems to screw up the Requests/urllib3 libs.
_DEFAULT_TIMEOUT_SEC = 30
_HMAC_HEADER = 'x-ycm-hmac'


class BaseRequest( object ):

  def __init__( self ):
    pass


  def Start( self ):
    pass


  def Done( self ):
    return True


  def Response( self ):
    return {}

  # This method blocks
  # |timeout| is num seconds to tolerate no response from server before giving
  # up; see Requests docs for details (we just pass the param along).
  @staticmethod
  def GetDataFromHandler( handler, timeout = _DEFAULT_TIMEOUT_SEC ):
    return JsonFromFuture( BaseRequest._TalkToHandlerAsync( '',
                                                            handler,
                                                            'GET',
                                                            timeout ) )


  # This is the blocking version of the method. See below for async.
  # |timeout| is num seconds to tolerate no response from server before giving
  # up; see Requests docs for details (we just pass the param along).
  @staticmethod
  def PostDataToHandler( data, handler, timeout = _DEFAULT_TIMEOUT_SEC ):
    return JsonFromFuture( BaseRequest.PostDataToHandlerAsync( data,
                                                               handler,
                                                               timeout ) )


  # This returns a future! Use JsonFromFuture to get the value.
  # |timeout| is num seconds to tolerate no response from server before giving
  # up; see Requests docs for details (we just pass the param along).
  @staticmethod
  def PostDataToHandlerAsync( data, handler, timeout = _DEFAULT_TIMEOUT_SEC ):
    return BaseRequest._TalkToHandlerAsync( data, handler, 'POST', timeout )


  # This returns a future! Use JsonFromFuture to get the value.
  # |method| is either 'POST' or 'GET'.
  # |timeout| is num seconds to tolerate no response from server before giving
  # up; see Requests docs for details (we just pass the param along).
  @staticmethod
  def _TalkToHandlerAsync( data,
                           handler,
                           method,
                           timeout = _DEFAULT_TIMEOUT_SEC ):
    def SendRequest( data, handler, method, timeout ):
      request_uri = _BuildUri( handler )
      if method == 'POST':
        sent_data = _ToUtf8Json( data )
        return BaseRequest.session.post(
            request_uri,
            data = sent_data,
            headers = BaseRequest._ExtraHeaders( method,
                                                 request_uri,
                                                 sent_data ),
            timeout = timeout )
      if method == 'GET':
        return BaseRequest.session.get(
            request_uri,
            headers = BaseRequest._ExtraHeaders( method, request_uri ),
            timeout = timeout )

    @retries( 5, delay = 0.5, backoff = 1.5 )
    def DelayedSendRequest( data, handler, method ):
      request_uri = _BuildUri( handler )
      if method == 'POST':
        sent_data = _ToUtf8Json( data )
        return requests.post(
            request_uri,
            data = sent_data,
            headers = BaseRequest._ExtraHeaders( method,
                                                 request_uri,
                                                 sent_data ) )
      if method == 'GET':
        return requests.get(
            request_uri,
            headers = BaseRequest._ExtraHeaders( method, request_uri ) )

    if not _CheckServerIsHealthyWithCache():
      return _EXECUTOR.submit( DelayedSendRequest, data, handler, method )

    return SendRequest( data, handler, method, timeout )


  @staticmethod
  def _ExtraHeaders( method, request_uri, request_body = None ):
    if not request_body:
      request_body = bytes( b'' )
    headers = dict( _HEADERS )
    headers[ _HMAC_HEADER ] = b64encode(
        CreateRequestHmac( ToBytes( method ),
                           ToBytes( urllib.parse.urlparse( request_uri ).path ),
                           request_body,
                           BaseRequest.hmac_secret ) )
    return headers

  session = FuturesSession( executor = _EXECUTOR )
  server_location = ''
  hmac_secret = ''


def BuildRequestData( filepath = None ):
  """Build request for the current buffer or the buffer corresponding to
  |filepath| if specified."""
  current_filepath = vimsupport.GetCurrentBufferFilepath()

  if filepath and current_filepath != filepath:
    # Cursor position is irrelevant when filepath is not the current buffer.
    return {
      'filepath': filepath,
      'line_num': 1,
      'column_num': 1,
      'file_data': vimsupport.GetUnsavedAndSpecifiedBufferData( filepath )
    }

  line, column = vimsupport.CurrentLineAndColumn()

  return {
    'filepath': current_filepath,
    'line_num': line + 1,
    'column_num': column + 1,
    'file_data': vimsupport.GetUnsavedAndSpecifiedBufferData( current_filepath )
  }


def JsonFromFuture( future ):
  response = future.result()
  _ValidateResponseObject( response )
  if response.status_code == requests.codes.server_error:
    raise MakeServerException( response.json() )

  # We let Requests handle the other status types, we only handle the 500
  # error code.
  response.raise_for_status()

  if response.text:
    return response.json()
  return None


def HandleServerException( exception, truncate = False ):
  serialized_exception = str( exception )

  # We ignore the exception about the file already being parsed since it comes
  # up often and isn't something that's actionable by the user.
  if 'already being parsed' in serialized_exception:
    return
  vimsupport.PostVimMessage( serialized_exception, truncate = truncate )


def _ToUtf8Json( data ):
  return ToBytes( json.dumps( data ) if data else None )


def _ValidateResponseObject( response ):
  our_hmac = CreateHmac( response.content, BaseRequest.hmac_secret )
  their_hmac = ToBytes( b64decode( response.headers[ _HMAC_HEADER ] ) )
  if not SecureBytesEqual( our_hmac, their_hmac ):
    raise RuntimeError( 'Received invalid HMAC for response!' )
  return True


def _BuildUri( handler ):
  return native( ToBytes( urllib.parse.urljoin( BaseRequest.server_location,
                                                handler ) ) )


SERVER_HEALTHY = False


def _CheckServerIsHealthyWithCache():
  global SERVER_HEALTHY

  def _ServerIsHealthy():
    request_uri = _BuildUri( 'healthy' )
    response = requests.get( request_uri,
                             headers = BaseRequest._ExtraHeaders(
                                 'GET', request_uri, bytes( b'' ) ) )
    _ValidateResponseObject( response )
    response.raise_for_status()
    return response.json()

  if SERVER_HEALTHY:
    return True

  try:
    SERVER_HEALTHY = _ServerIsHealthy()
    return SERVER_HEALTHY
  except:
    return False


def MakeServerException( data ):
  if data[ 'exception' ][ 'TYPE' ] == UnknownExtraConf.__name__:
    return UnknownExtraConf( data[ 'exception' ][ 'extra_conf_file' ] )

  return ServerError( '{0}: {1}'.format( data[ 'exception' ][ 'TYPE' ],
                                         data[ 'message' ] ) )