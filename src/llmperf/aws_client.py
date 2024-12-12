import boto3
from botocore.credentials import AssumeRoleCredentialFetcher, DeferredRefreshableCredentials
from botocore.session import Session
import logging
LOGGER = logging.getLogger(__name__)


def get_current_user(session:boto3.Session):
    sts_client = session.client('sts')
    identity = sts_client.get_caller_identity()
  
    arn = identity.get('Arn', '')
    user = arn.split('/')[1]
    return user


class AWSSession(boto3.Session):

    def __init__(
            self,
            role_arn, 
            role_session_name,
            custom_region_name=None,
            refresh_in=AssumeRoleCredentialFetcher.DEFAULT_EXPIRY_WINDOW_SECONDS
        ) -> None:

        super().__init__()
        self.role_arn = role_arn
        self.role_session_name = role_session_name
        self.refresh_in = refresh_in

        self.custom_region_name = custom_region_name
        # Create an initial STS client to assume the role
        self.sts_client = self.client("sts", region_name=self.custom_region_name)
        self._create_refreshable_sts_session()

    def __del__(self):
        if self._session:
            del self._session

    def _create_refreshable_sts_session(self):
        # Base session
        session = Session()
        if self.custom_region_name:
            session.set_config_variable("region", self.custom_region_name)
            LOGGER.info(f'Changed region name to {self.region_name}')

        duration_seconds = AssumeRoleCredentialFetcher.DEFAULT_EXPIRY_WINDOW_SECONDS + self.refresh_in
        # Credential fetching using STS AssumeRole
        fetcher = AssumeRoleCredentialFetcher(
            client_creator=session.create_client,
            source_credentials=session.get_credentials(),
            role_arn=self.role_arn,
            extra_args={'RoleSessionName': self.role_session_name, 'DurationSeconds': duration_seconds},
        )

        # Deferred credentials that refresh automatically
        refreshable_credentials = DeferredRefreshableCredentials(
            fetcher.fetch_credentials,
            'assume-role'
        )
        session._credentials = refreshable_credentials

        # Update botocore session in boto3.Session
        self._session = session
