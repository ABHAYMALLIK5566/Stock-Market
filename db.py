from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()
engine = create_engine('sqlite:///marketbot.db', echo=False)
Session = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    phone = Column(String, unique=True, nullable=False)
    watchlist = relationship('Watchlist', back_populates='user', cascade='all, delete-orphan')
    alerts = relationship('Alert', back_populates='user', cascade='all, delete-orphan')
    subscription = relationship('Subscription', back_populates='user', uselist=False, cascade='all, delete-orphan')

class Watchlist(Base):
    __tablename__ = 'watchlists'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String, nullable=False)
    user = relationship('User', back_populates='watchlist')
    __table_args__ = (UniqueConstraint('user_id', 'symbol', name='_user_symbol_uc'),)

class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    triggered = Column(Boolean, default=False)
    user = relationship('User', back_populates='alerts')

class Subscription(Base):
    __tablename__ = 'subscriptions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    daily = Column(Boolean, default=False)
    user = relationship('User', back_populates='subscription')

Base.metadata.create_all(engine)

def get_or_create_user(phone):
    session = Session()
    user = session.query(User).filter_by(phone=phone).first()
    if not user:
        user = User(phone=phone)
        session.add(user)
        session.commit()
    session.close()
    return user

def add_watchlist(phone, symbol):
    session = Session()
    user = session.query(User).filter_by(phone=phone).first()
    if not user:
        user = User(phone=phone)
        session.add(user)
        session.commit()
    if not session.query(Watchlist).filter_by(user_id=user.id, symbol=symbol).first():
        wl = Watchlist(user_id=user.id, symbol=symbol)
        session.add(wl)
        session.commit()
    session.close()

def remove_watchlist(phone, symbol):
    session = Session()
    user = session.query(User).filter_by(phone=phone).first()
    if user:
        wl = session.query(Watchlist).filter_by(user_id=user.id, symbol=symbol).first()
        if wl:
            session.delete(wl)
            session.commit()
    session.close()

def get_watchlist(phone):
    session = Session()
    user = session.query(User).filter_by(phone=phone).first()
    symbols = []
    if user:
        symbols = [w.symbol for w in user.watchlist]
    session.close()
    return symbols

def add_alert(phone, symbol, price):
    session = Session()
    user = session.query(User).filter_by(phone=phone).first()
    if not user:
        user = User(phone=phone)
        session.add(user)
        session.commit()
    alert = Alert(user_id=user.id, symbol=symbol, price=price, triggered=False)
    session.add(alert)
    session.commit()
    session.close()

def get_alerts():
    session = Session()
    alerts = session.query(Alert).filter_by(triggered=False).all()
    result = []
    for a in alerts:
        result.append({'phone': a.user.phone, 'symbol': a.symbol, 'price': a.price, 'id': a.id})
    session.close()
    return result

def mark_alert_triggered(alert_id):
    session = Session()
    alert = session.query(Alert).filter_by(id=alert_id).first()
    if alert:
        alert.triggered = True
        session.commit()
    session.close()

def subscribe_daily(phone):
    session = Session()
    user = session.query(User).filter_by(phone=phone).first()
    if not user:
        user = User(phone=phone)
        session.add(user)
        session.commit()
    sub = session.query(Subscription).filter_by(user_id=user.id).first()
    if not sub:
        sub = Subscription(user_id=user.id, daily=True)
        session.add(sub)
    else:
        sub.daily = True
    session.commit()
    session.close()

def unsubscribe_daily(phone):
    session = Session()
    user = session.query(User).filter_by(phone=phone).first()
    if user:
        sub = session.query(Subscription).filter_by(user_id=user.id).first()
        if sub:
            sub.daily = False
            session.commit()
    session.close()

def get_daily_subscribers():
    session = Session()
    subs = session.query(Subscription).filter_by(daily=True).all()
    phones = [s.user.phone for s in subs]
    session.close()
    return phones 