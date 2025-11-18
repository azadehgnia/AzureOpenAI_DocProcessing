"""Azure SQL incident ingestion/query using SQLAlchemy
"""

import os
import struct
from typing import List, Optional

from pydantic import BaseModel
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

import pyodbc  # still used by SQLAlchemy creator for Azure AD token flow
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    func,
)
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker

# Load environment variables early
load_dotenv()

Base = declarative_base()

# Load environment variables
load_dotenv()

# Pydantic Models
class IncidentHeader(BaseModel):
    documentId: str
    pageCount: str
    IncidentDate: str
    Category: str
    NumberOfImpactedEmployees: int
    EmployeeNames: str
    Location: str
    IncidentType: str
    Injuries: str
    Summary: str


class EmployeeImpact(BaseModel):
    EmployeeName: str
    EmployeeID: str
    InjuryDescription: str
    ActionTaken: str


class Incident(BaseModel):
    header: IncidentHeader
    impact_details: List[EmployeeImpact]


# SQLAlchemy ORM models -------------------------------------------------------

class IncidentHeaderORM(Base):
    __tablename__ = "IncidentHeader"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    documentId = Column(String(255), unique=True, nullable=False, index=True)
    pageCount = Column(String(50))
    IncidentDate = Column(String(50))  # Could convert to DATE if format known
    Category = Column(String(100), index=True)
    NumberOfImpactedEmployees = Column(Integer)
    EmployeeNames = Column(String)  # NVARCHAR(MAX) approximated by Text/String
    Location = Column(String(255))
    IncidentType = Column(String(100))
    Injuries = Column(String)
    Summary = Column(String)
    CreatedDate = Column(DateTime, server_default=func.getdate())

    impacts = relationship(
        "EmployeeImpactORM",
        back_populates="incident",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class EmployeeImpactORM(Base):
    __tablename__ = "EmployeeImpact"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    documentId = Column(String(255), ForeignKey("IncidentHeader.documentId", ondelete="CASCADE"), nullable=False, index=True)
    EmployeeName = Column(String(255))
    EmployeeID = Column(String(50))
    InjuryDescription = Column(String)
    ActionTaken = Column(String)
    CreatedDate = Column(DateTime, server_default=func.getdate())

    incident = relationship("IncidentHeaderORM", back_populates="impacts")


class AzureSQLDatabase:
    """Azure SQL Database access layer using SQLAlchemy ORM."""

    def __init__(
        self,
        server: str,
        database: str,
        use_azure_auth: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver: str = "ODBC Driver 18 for SQL Server",
    ):
        self.server = server
        self.database = database
        self.use_azure_auth = use_azure_auth
        self.username = username
        self.password = password
        self.driver = driver
        self._engine = None
        self._SessionLocal: Optional[sessionmaker] = None

    # ------------------------------------------------------------------
    def _create_engine(self):
        """Create SQLAlchemy engine with either Azure AD or SQL Auth."""
        # Base ODBC connection string (DSN-less)
        odbc_str_base = (
            f"Driver={{{self.driver}}};Server={self.server};Database={self.database};"  # braces needed around driver for ODBC
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        )

        if self.use_azure_auth:
            credential = DefaultAzureCredential()

            def get_conn():
                token = credential.get_token("https://database.windows.net/.default")
                token_bytes = token.token.encode("UTF-16-LE")
                token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
                SQL_COPT_SS_ACCESS_TOKEN = 1256
                return pyodbc.connect(
                    odbc_str_base,
                    attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct},
                )

            # Use creator pattern (no URL credentials needed)
            self._engine = create_engine(
                "mssql+pyodbc://", creator=get_conn, fast_executemany=True, future=True
            )
        else:
            # SQL Authentication via URL
            if not self.username or not self.password:
                raise ValueError("SQL authentication requires username and password.")
            # Proper URL encoding of driver spaces -> +
            url = (
                f"mssql+pyodbc://{self.username}:{self.password}@{self.server}/{self.database}?"
                f"driver={self.driver.replace(' ', '+')}"
                "&Encrypt=yes&TrustServerCertificate=no"
            )
            self._engine = create_engine(url, fast_executemany=True, future=True)

        self._SessionLocal = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)

    # ------------------------------------------------------------------
    def connect(self):
        if not self._engine:
            self._create_engine()
            print(f"SQLAlchemy engine created for database '{self.database}'.")
        # Test connection
        with self._engine.connect() as conn:
            conn.execute(func.now().select()) if hasattr(func.now(), 'select') else None
        return self._engine

    # ------------------------------------------------------------------
    def create_tables(self):
        if not self._engine:
            raise RuntimeError("Engine not initialized. Call connect() first.")
        Base.metadata.create_all(self._engine)
        print("ORM tables ensured (create if not exists).")

    # ------------------------------------------------------------------
    def _session(self) -> Session:
        if not self._SessionLocal:
            raise RuntimeError("Session factory not initialized. Call connect() first.")
        return self._SessionLocal()

    # ------------------------------------------------------------------
    def insert_incident(self, incident: Incident):
        session = self._session()
        try:
            # Check existing
            existing = (
                session.query(IncidentHeaderORM)
                .filter(IncidentHeaderORM.documentId == incident.header.documentId)
                .one_or_none()
            )
            if existing:
                raise ValueError(
                    f"Incident with documentId '{incident.header.documentId}' already exists."
                )

            header_orm = IncidentHeaderORM(
                documentId=incident.header.documentId,
                pageCount=incident.header.pageCount,
                IncidentDate=incident.header.IncidentDate,
                Category=incident.header.Category,
                NumberOfImpactedEmployees=incident.header.NumberOfImpactedEmployees,
                EmployeeNames=incident.header.EmployeeNames,
                Location=incident.header.Location,
                IncidentType=incident.header.IncidentType,
                Injuries=incident.header.Injuries,
                Summary=incident.header.Summary,
            )

            for emp in incident.impact_details:
                header_orm.impacts.append(
                    EmployeeImpactORM(
                        documentId=incident.header.documentId,
                        EmployeeName=emp.EmployeeName,
                        EmployeeID=emp.EmployeeID,
                        InjuryDescription=emp.InjuryDescription,
                        ActionTaken=emp.ActionTaken,
                    )
                )

            session.add(header_orm)
            session.commit()
            print(f"Incident '{incident.header.documentId}' inserted successfully (ORM).")
        except Exception as e:
            session.rollback()
            print(f"Error inserting incident: {e}")
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    def query_all_incidents(self):
        session = self._session()
        try:
            rows = (
                session.query(IncidentHeaderORM)
                .order_by(IncidentHeaderORM.CreatedDate.desc())
                .all()
            )
            return [
                {
                    "documentId": r.documentId,
                    "pageCount": r.pageCount,
                    "IncidentDate": r.IncidentDate,
                    "Category": r.Category,
                    "NumberOfImpactedEmployees": r.NumberOfImpactedEmployees,
                    "EmployeeNames": r.EmployeeNames,
                    "Location": r.Location,
                    "IncidentType": r.IncidentType,
                    "Injuries": r.Injuries,
                    "Summary": r.Summary,
                    "CreatedDate": r.CreatedDate,
                }
                for r in rows
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    def query_incident_by_id(self, document_id: str):
        session = self._session()
        try:
            header = (
                session.query(IncidentHeaderORM)
                .filter(IncidentHeaderORM.documentId == document_id)
                .one_or_none()
            )
            if not header:
                return None
            return {
                "header": {
                    "documentId": header.documentId,
                    "pageCount": header.pageCount,
                    "IncidentDate": header.IncidentDate,
                    "Category": header.Category,
                    "NumberOfImpactedEmployees": header.NumberOfImpactedEmployees,
                    "EmployeeNames": header.EmployeeNames,
                    "Location": header.Location,
                    "IncidentType": header.IncidentType,
                    "Injuries": header.Injuries,
                    "Summary": header.Summary,
                },
                "impact_details": [
                    {
                        "EmployeeName": i.EmployeeName,
                        "EmployeeID": i.EmployeeID,
                        "InjuryDescription": i.InjuryDescription,
                        "ActionTaken": i.ActionTaken,
                    }
                    for i in header.impacts
                ],
            }
        finally:
            session.close()

    # ------------------------------------------------------------------
    def query_incidents_by_category(self, category: str):
        session = self._session()
        try:
            rows = (
                session.query(IncidentHeaderORM)
                .filter(IncidentHeaderORM.Category == category)
                .order_by(IncidentHeaderORM.IncidentDate.desc())
                .all()
            )
            return [
                {
                    "documentId": r.documentId,
                    "pageCount": r.pageCount,
                    "IncidentDate": r.IncidentDate,
                    "Category": r.Category,
                    "NumberOfImpactedEmployees": r.NumberOfImpactedEmployees,
                    "EmployeeNames": r.EmployeeNames,
                    "Location": r.Location,
                    "IncidentType": r.IncidentType,
                    "Injuries": r.Injuries,
                    "Summary": r.Summary,
                }
                for r in rows
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    def query_employee_impacts(self, employee_name: str = None):
        session = self._session()
        try:
            q = session.query(EmployeeImpactORM, IncidentHeaderORM).join(
                IncidentHeaderORM,
                EmployeeImpactORM.documentId == IncidentHeaderORM.documentId,
            )
            if employee_name:
                q = q.filter(EmployeeImpactORM.EmployeeName.ilike(f"%{employee_name}%"))
            rows = q.order_by(IncidentHeaderORM.IncidentDate.desc()).all()
            return [
                {
                    "documentId": ei.documentId,
                    "EmployeeName": ei.EmployeeName,
                    "EmployeeID": ei.EmployeeID,
                    "InjuryDescription": ei.InjuryDescription,
                    "ActionTaken": ei.ActionTaken,
                    "IncidentDate": ih.IncidentDate,
                    "Location": ih.Location,
                    "IncidentType": ih.IncidentType,
                }
                for (ei, ih) in rows
            ]
        finally:
            session.close()


# Example usage
def main():
    """Demonstrate ORM usage."""

    server = os.getenv("AZURE_SQL_SERVER")
    database = os.getenv("AZURE_SQL_DATABASE")
    use_azure_auth = os.getenv("USE_AZURE_AUTH", "true").lower() == "true"
    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")

    db = AzureSQLDatabase(
        server=server,
        database=database,
        use_azure_auth=use_azure_auth,
        username=username,
        password=password,
    )

    db.connect()
    db.create_tables()

    sample_incident = Incident(
        header=IncidentHeader(
            documentId="INC-2025-001",
            pageCount="3",
            IncidentDate="2025-11-15",
            Category="Workplace Safety",
            NumberOfImpactedEmployees=2,
            EmployeeNames="John Doe, Jane Smith",
            Location="Manufacturing Plant A",
            IncidentType="Equipment Malfunction",
            Injuries="Minor cuts and bruises",
            Summary="Equipment malfunction resulted in minor injuries to two employees",
        ),
        impact_details=[
            EmployeeImpact(
                EmployeeName="John Doe",
                EmployeeID="EMP001",
                InjuryDescription="Minor cut on left hand",
                ActionTaken="First aid administered, sent to medical center for evaluation",
            ),
            EmployeeImpact(
                EmployeeName="Jane Smith",
                EmployeeID="EMP002",
                InjuryDescription="Bruise on right arm",
                ActionTaken="Ice pack applied, monitoring for 24 hours",
            ),
        ],
    )

    try:
        db.insert_incident(sample_incident)
    except ValueError:
        print("Sample incident already exists; continuing.")

    print("\n=== All Incidents ===")
    for inc in db.query_all_incidents():
        print(f"ID: {inc['documentId']} Date: {inc['IncidentDate']} Category: {inc['Category']}")

    print("\n=== Specific Incident ===")
    spec = db.query_incident_by_id("INC-2025-001")
    if spec:
        print(f"Summary: {spec['header']['Summary']}")
        print(f"Impacts: {len(spec['impact_details'])}")

    print("\n=== Incidents by Category ===")
    cat_rows = db.query_incidents_by_category("Workplace Safety")
    print(f"Found {len(cat_rows)} incidents in category.")

    print("\n=== Employee Impacts (filter John) ===")
    for imp in db.query_employee_impacts("John"):
        print(f"Employee: {imp['EmployeeName']} Injury: {imp['InjuryDescription']}")


if __name__ == "__main__":
    main()
